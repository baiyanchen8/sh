# ============================================================
# CIFAR-10 + CapsuleBD Attack + FLTrust Aggregation (Single File)
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
LOCAL_EPOCHS = 3
BATCH_SIZE = 128
NUM_WORKERS = 2

LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 3000
REF_SIZE = 2000

# CapsuleBD parameters
POISON_RATE = 0.2
LAMBDA_POISON = 0.7
LAMBDA_SHELL = 0.3

# Trigger
TRIGGER_SIZE = 4
TARGET_LABEL = 0

# Server warmup (ONLY early rounds)
WARMUP_ROUNDS = 5
WARMUP_STEPS = 5
WARMUP_LR = 5e-3

# FLTrust server update steps
G0_STEPS = 5

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
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)
    return [Subset(dataset, idx[i*per_client:(i+1)*per_client]) for i in range(n)]

client_datasets = split_clients(trainset, CIFAR_SUBSET_PER_CLIENT, NUM_CLIENTS)

test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# Use a clean reference set (kept as testset here to match your earlier scripts)
ref_subset = Subset(testset, list(range(min(REF_SIZE, len(testset)))))
ref_loader = DataLoader(ref_subset, batch_size=256, shuffle=True, num_workers=NUM_WORKERS)

# ---------------- TRIGGER (normalized white patch) ----------------
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD_T  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

def add_trigger(x: torch.Tensor):
    x = x.clone()
    ts = int(TRIGGER_SIZE)
    _, h, w = x.shape
    y0 = int(h) - ts
    x0 = int(w) - ts
    white = (1.0 - IMAGENET_MEAN_T.to(x.device)) / IMAGENET_STD_T.to(x.device)  # [3,1,1]
    x[:, y0:y0 + ts, x0:x0 + ts] = white
    return x

# ---------------- MODEL ----------------
def make_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # freeze all
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze last block + head
    for n, p in model.named_parameters():
        if n.startswith("layer4") or n.startswith("fc"):
            p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def make_opt(model, lr=LR_HEAD):
    return optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

# ---------------- LOCAL TRAINING ----------------
def train_benign_from_global(global_model, loader, epochs=LOCAL_EPOCHS):
    m = deepcopy(global_model)
    opt = make_opt(m, lr=LR_HEAD)
    m.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            opt.step()
    return m.state_dict()

def train_capsulebd(global_model, loader):
    poison = deepcopy(global_model)
    shell  = deepcopy(global_model)

    opt_p = make_opt(poison, lr=LR_HEAD)
    opt_s = make_opt(shell,  lr=LR_HEAD)

    poison.train(); shell.train()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            xp = x.clone()
            yp = y.clone()
            for i in range(len(xp)):
                if random.random() < POISON_RATE:
                    xp[i] = add_trigger(xp[i])
                    yp[i] = TARGET_LABEL

            opt_p.zero_grad()
            loss_p = criterion(poison(xp), yp)
            loss_p.backward()
            opt_p.step()

            opt_s.zero_grad()
            loss_s = criterion(shell(x), y)
            loss_s.backward()
            opt_s.step()

    new_state = {}
    gp = poison.state_dict()
    gs = shell.state_dict()
    for k in global_model.state_dict().keys():
        new_state[k] = (LAMBDA_POISON * gp[k] + LAMBDA_SHELL * gs[k]).detach()
    return new_state

# ---------------- EVALUATION ----------------
@torch.no_grad()
def eval_acc(model):
    model.eval()
    c = t = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = model(x).argmax(1)
        c += (p == y).sum().item()
        t += y.size(0)
    return 100.0 * c / max(1, t)

@torch.no_grad()
def eval_asr(model):
    model.eval()
    hit = tot = 0
    for x, _ in test_loader:
        x = x.to(DEVICE)
        xt = torch.stack([add_trigger(img) for img in x]).to(DEVICE)
        pred = model(xt).argmax(1)
        hit += (pred == TARGET_LABEL).sum().item()
        tot += pred.size(0)
    return 100.0 * hit / max(1, tot)

# ---------------- FLTRUST ----------------
def state_to_vec(sd):
    return torch.cat([v.detach().flatten().float().cpu() for v in sd.values()])

def dict_delta(new_sd, base_sd):
    return {k: (new_sd[k].detach() - base_sd[k].detach()) for k in base_sd.keys()}

def model_update_delta(base_model, loader, steps=G0_STEPS):
    """Return delta dict: (trained_model - base_model) after a few steps on loader."""
    m = deepcopy(base_model)
    opt = make_opt(m, lr=LR_HEAD)
    m.train()
    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = criterion(m(x), y)
        loss.backward()
        opt.step()
    return dict_delta(m.state_dict(), base_model.state_dict())

def fltrust_aggregate(client_updates, g0, eps=1e-12, min_norm=1e-6, max_scale=10.0, min_Z=1e-8):
    """
    Paper-faithful FLTrust with a necessary numerical guard:
    - skip tiny/non-finite updates (prevents sudden collapse)
    - clamp scale (prevents exploding update if ||gi|| is tiny)
    """
    g0v = state_to_vec(g0)
    n0 = torch.norm(g0v).item()
    if (not math.isfinite(n0)) or (n0 < min_norm):
        return None

    weights, scaled = [], []
    for gi in client_updates:
        giv = state_to_vec(gi)
        ngi = torch.norm(giv).item()
        if (not math.isfinite(ngi)) or (ngi < min_norm):
            continue

        cos = (torch.dot(giv, g0v).item()) / (ngi * n0 + eps)
        if not math.isfinite(cos):
            continue

        ts = max(0.0, cos)
        if ts <= 0.0:
            continue

        scale = n0 / (ngi + eps)
        if not math.isfinite(scale):
            continue
        scale = min(scale, max_scale)

        weights.append(ts)
        scaled.append({k: v * scale for k, v in gi.items()})

    Z = sum(weights)
    if (not math.isfinite(Z)) or (Z < min_Z):
        return None

    agg = {}
    for k in g0.keys():
        agg[k] = sum(weights[i] * scaled[i][k] for i in range(len(weights))) / Z
    return agg

# ---------------- FEDERATED LOOP ----------------
global_model = make_model()

metrics = {
    "round": [],
    "acc": [],
    "asr": [],
}

for r in range(1, ROUNDS + 1):
    print(f"\n=== Round {r}/{ROUNDS} ===")
    t0_round = time.time()

    # ---- warmup ONLY in early rounds ----
    if r <= WARMUP_ROUNDS:
        global_model.train()
        opt_warm = make_opt(global_model, lr=WARMUP_LR)
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

    # ---- base snapshot ----
    base_sd = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

    # ---- server reference update g0 (delta) ----
    g0 = model_update_delta(global_model, ref_loader, steps=G0_STEPS)

    # ---- client updates (deltas) ----
    client_updates = []
    t_inf_sum = 0.0

    for cid in range(NUM_CLIENTS):
        loader = DataLoader(
            client_datasets[cid],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

        if cid == ATTACKER_IDX:
            sd = train_capsulebd(global_model, loader)  # full state
        else:
            sd = train_benign_from_global(global_model, loader, epochs=LOCAL_EPOCHS)

        gi = dict_delta(sd, base_sd)
        client_updates.append(gi)

        

    # ---- FLTrust aggregation ----
    g = fltrust_aggregate(client_updates, g0)

    if g is not None:
        new_sd = {}
        for k in base_sd.keys():
            new_sd[k] = base_sd[k] + g[k].to(base_sd[k].device).type_as(base_sd[k])

        # safety: skip if any non-finite appears
        bad = False
        for v in new_sd.values():
            if not torch.isfinite(v).all():
                bad = True
                break
        if bad:
            print("WARNING: Non-finite global weights detected. Skipping update this round.")
        else:
            global_model.load_state_dict(new_sd)
    else:
        print("FLTrust: skipped update (no trusted clients or unstable norms).")

    # ---- evaluation ----
    acc = eval_acc(global_model)
    asr = eval_asr(global_model)

    print(f"ACC={acc:.2f}% | ASR={asr:.2f}%")

    metrics["round"].append(r)
    metrics["acc"].append(acc)
    metrics["asr"].append(asr)

# ---------------- SAVE ----------------
os.makedirs("outputs", exist_ok=True)
with open("outputs/capsulebd_cifar_fltrust.json", "w") as f:
    json.dump(metrics, f, indent=2)

torch.save(global_model.state_dict(), "outputs/capsulebd_cifar_fltrust.pth")

print("\nSaved outputs/capsulebd_cifar_fltrust.json and outputs/capsulebd_cifar_fltrust.pth")
