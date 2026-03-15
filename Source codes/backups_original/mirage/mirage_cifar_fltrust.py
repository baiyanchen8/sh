# ============================================================
# CIFAR-10 + MIRAGE Attack + FLTrust Aggregation (FIXED)
# ============================================================

import os, time, json, random, math
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

NUM_CLIENTS = 3
ATTACKER_IDX = 2
ROUNDS = 20
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
NUM_WORKERS = 2
ATTACKER_EPOCHS = 2

LR_HEAD = 3e-4
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 1500
REF_SIZE = 1000

# MIRAGE parameters
POISON_RATE_ATTACKER = 0.3
BD_LAMBDA = 2.0
TRIGGER_LR = 1e-1
TRIGGER_SIZE = 4
TARGET_LABEL = 0

# FLTrust
G0_STEPS = 5

# Warmup
WARMUP_ROUNDS = 5
WARMUP_STEPS = 5
WARMUP_LR = 5e-3

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- TRANSFORMS ----------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------- DATA ----------------
trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

def split_clients(dataset, per_client, n):
    idx = np.random.permutation(len(dataset))
    return [Subset(dataset, idx[i*per_client:(i+1)*per_client]) for i in range(n)]

client_datasets = split_clients(trainset, CIFAR_SUBSET_PER_CLIENT, NUM_CLIENTS)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
ref_subset = Subset(testset, list(range(REF_SIZE)))
ref_loader = DataLoader(ref_subset, batch_size=256, shuffle=True, num_workers=NUM_WORKERS)

# ---------------- MIRAGE TRIGGER -----------------
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(1,3,1,1)
IMAGENET_STD_T  = torch.tensor(IMAGENET_STD,  device=DEVICE).view(1,3,1,1)

def init_trigger():
    # normalized white value for each channel
    white = (1.0 - IMAGENET_MEAN_T) / IMAGENET_STD_T  # shape [1,3,1,1]
    trig = white.repeat(1, 1, TRIGGER_SIZE, TRIGGER_SIZE).squeeze(0).detach()
    return torch.nn.Parameter(trig, requires_grad=False)

def apply_trigger(x, trigger):
    x = x.clone()
    x[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = trigger
    return x


# ---------------- MODEL ----------------
def make_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    for n, p in m.named_parameters():
        if n.startswith("layer4") or n.startswith("fc"):
            p.requires_grad = True
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m.to(DEVICE)

criterion = nn.CrossEntropyLoss()

def make_opt(m, lr):
    return optim.SGD(filter(lambda p: p.requires_grad, m.parameters()),
                     lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)

# ---------------- MIRAGE TRAINING ----------------
def train_mirage(global_model, loader, trigger):
    m = deepcopy(global_model)
    opt_m = make_opt(m, LR_HEAD)
    opt_t = optim.Adam([trigger], lr=TRIGGER_LR)

    m.train()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            mask = torch.rand(len(x), device=DEVICE) < POISON_RATE_ATTACKER
            loss = 0.0

            if (~mask).any():
                loss += criterion(m(x[~mask]), y[~mask])

            if mask.any():
                xt = apply_trigger(x[mask], trigger)
                yt = torch.full((mask.sum(),), TARGET_LABEL,
                                device=DEVICE, dtype=torch.long)
                loss += BD_LAMBDA * criterion(m(xt), yt)

            opt_m.zero_grad()
            opt_t.zero_grad()
            loss.backward()
            opt_m.step()
            opt_t.step()

            with torch.no_grad():
                trigger.clamp_(-3, 3)

    return m.state_dict()

def train_benign(global_model, loader):
    m = deepcopy(global_model)
    opt = make_opt(m, LR_HEAD)
    m.train()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            opt.step()
    return m.state_dict()

# ---------------- FLTRUST ----------------
def state_to_vec(sd):
    return torch.cat([v.flatten().float().cpu() for v in sd.values()])

def dict_delta(new_sd, base_sd):
    return {k: new_sd[k] - base_sd[k] for k in base_sd}

def model_update_delta(base_model, loader, steps):
    m = deepcopy(base_model)
    opt = make_opt(m, LR_HEAD)
    m.train()
    it = iter(loader)
    for _ in range(steps):
        try: x, y = next(it)
        except StopIteration:
            it = iter(loader); x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(m(x), y)
        loss.backward()
        opt.step()
    return dict_delta(m.state_dict(), base_model.state_dict())

def fltrust_aggregate(client_updates, g0):
    g0v = state_to_vec(g0)
    n0 = torch.norm(g0v) + 1e-12
    weights, scaled = [], []

    for gi in client_updates:
        giv = state_to_vec(gi)
        ngi = torch.norm(giv)
        if ngi < 1e-6: continue
        cos = torch.dot(giv, g0v) / (ngi * n0 + 1e-12)
        ts = max(0.0, cos.item())
        if ts == 0: continue
        scale = min(n0 / (ngi + 1e-12), 10.0)
        weights.append(ts)
        scaled.append({k: v * scale for k, v in gi.items()})

    Z = sum(weights)
    if Z == 0: return None
    return {k: sum(weights[i] * scaled[i][k] for i in range(len(weights))) / Z
            for k in g0}

# ---------------- EVAL ----------------
@torch.no_grad()
def eval_acc(m):
    m.eval()
    c = t = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        c += (m(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return 100 * c / t

@torch.no_grad()
def eval_asr(m, trigger):
    m.eval()
    hit = tot = 0
    for x, y in test_loader:
        mask = torch.ones_like(y, dtype=torch.bool)
        if mask.any():
            xt = apply_trigger(x[mask].to(DEVICE), trigger)
            hit += (m(xt).argmax(1) == TARGET_LABEL).sum().item()
            tot += mask.sum().item()
    return 100 * hit / max(1, tot)

# ---------------- FEDERATED LOOP ----------------
global_model = make_model()
trigger = init_trigger()

metrics = {"round":[], "acc":[], "asr":[]}
WARMUP_ROUNDS = 3       
WARMUP_STEPS  = 5        
WARMUP_LR     = 5e-3

for r in range(1, ROUNDS+1):
    print(f"\n=== Round {r}/{ROUNDS} ===")

    for r in range(1, ROUNDS + 1):

    
        if r <= WARMUP_ROUNDS:
            global_model.train()
            opt_warm = torch.optim.SGD(
                [p for p in global_model.parameters() if p.requires_grad],
                lr=WARMUP_LR,
                momentum=0.9,
                weight_decay=WEIGHT_DECAY
            )

            it = iter(ref_loader)
            for _ in range(WARMUP_STEPS):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(ref_loader)
                    x, y = next(it)

                x, y = x.to(DEVICE), y.to(DEVICE)
                opt_warm.zero_grad()
                loss = criterion(global_model(x), y)
                loss.backward()
                opt_warm.step()
   


    base_sd = deepcopy(global_model.state_dict())
    g0 = model_update_delta(global_model, ref_loader, G0_STEPS)

    client_updates = []
    for cid in range(NUM_CLIENTS):
        loader = DataLoader(client_datasets[cid], batch_size=BATCH_SIZE, shuffle=True)
        sd = train_mirage(global_model, loader, trigger) if cid == ATTACKER_IDX \
             else train_benign(global_model, loader)
        client_updates.append(dict_delta(sd, base_sd))

    g = fltrust_aggregate(client_updates, g0)
    if g:
        global_model.load_state_dict({k: base_sd[k] + g[k].to(DEVICE) for k in base_sd})

    acc = eval_acc(global_model)
    asr = eval_asr(global_model, trigger)
    print(f"ACC={acc:.2f}% | ASR={asr:.2f}%")

    metrics["round"].append(r)
    metrics["acc"].append(acc)
    metrics["asr"].append(asr)

# ---------------- SAVE ----------------
os.makedirs("outputs", exist_ok=True)
json.dump(metrics, open("outputs/cifar_mirage_fltrust.json","w"), indent=2)
torch.save(global_model.state_dict(), "outputs/cifar_mirage_fltrust.pth")
print("\nSaved outputs.")
