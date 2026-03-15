import os, time, json, random, copy
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset

# ======================================================
# CONFIG
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLIENTS = 4
ATTACKER_IDX = [2, 3]

ROUNDS = 20
LOCAL_EPOCHS = 2               # CapsuleBD default
BATCH_SIZE = 128
NUM_WORKERS = 2

LR = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 3000

# CapsuleBD parameters (from your file)
POISON_RATE = 0.20
TARGET_LABEL = 0
TRIGGER_SIZE = 4
TRIGGER_ALPHA = 0.25

LAMBDA_POISON = 0.7
LAMBDA_SHELL  = 0.3

# Metrics / logging (kept same as your FedAvg baseline)
REF_SIZE = 2000
SAVE_PATH = "outputs/capsulebd_cifar_fedavg_metrics.json"

# ======================================================
# DATA
# ======================================================
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

NORMALIZER = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

def make_client_subsets(dataset, per_client, n_clients):
    indices = list(range(min(len(dataset), per_client * n_clients)))
    return [Subset(dataset, indices[i*per_client:(i+1)*per_client]) for i in range(n_clients)]

client_datasets = make_client_subsets(trainset, CIFAR_SUBSET_PER_CLIENT, NUM_CLIENTS)
client_loaders = [
    DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    for ds in client_datasets
]

test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# ======================================================
# SERVER REFERENCE SET (metrics only)
# ======================================================
trainset_ref = datasets.CIFAR10("./data", train=True, download=True, transform=transform_test)

all_idx = list(range(len(trainset_ref)))
random.shuffle(all_idx)

ref_idx = []
cls_cnt = {i: 0 for i in range(10)}
per_class = REF_SIZE // 10

for idx in all_idx:
    _, y = trainset_ref[idx]
    if cls_cnt[y] < per_class:
        ref_idx.append(idx)
        cls_cnt[y] += 1
    if len(ref_idx) >= REF_SIZE:
        break

ref_loader = DataLoader(
    Subset(trainset_ref, ref_idx),
    batch_size=256,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# ======================================================
# MODEL
# ======================================================
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for name, p in model.named_parameters():
        p.requires_grad = ("layer4" in name or "fc" in name)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

global_model = build_model()
criterion = nn.CrossEntropyLoss()

def make_optimizer(model):
    return optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

# ======================================================
# CAPSULEBD TRIGGER (blended, from your file)
# ======================================================
def add_capsulebd_trigger(x):
    x = x.clone()
    _, h, w = x.shape
    patch = x[:, h-TRIGGER_SIZE:h, w-TRIGGER_SIZE:w]
    x[:, h-TRIGGER_SIZE:h, w-TRIGGER_SIZE:w] = (
        (1 - TRIGGER_ALPHA) * patch + TRIGGER_ALPHA
    )
    return torch.clamp(x, 0, 1)

# ======================================================
# CAPSULEBD LOCAL TRAINING (ATTACKER)
# ======================================================
def train_capsulebd_local(global_model, loader):
    model_poison = copy.deepcopy(global_model)
    model_shell  = copy.deepcopy(global_model)

    opt_p = make_optimizer(model_poison)
    opt_s = make_optimizer(model_shell)

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = NORMALIZER(x)

            # ----- poison model -----
            xp = x.clone()
            yp = y.clone()
            for i in range(len(xp)):
                if random.random() < POISON_RATE:
                    xp[i] = add_capsulebd_trigger(xp[i])
                    yp[i] = TARGET_LABEL

            xp = NORMALIZER(xp)
            opt_p.zero_grad()
            loss_p = criterion(model_poison(xp), yp)
            loss_p.backward()
            opt_p.step()

            # ----- shell model -----
            x_norm = NORMALIZER(x.clone())
            opt_s.zero_grad()
            loss_s = criterion(model_shell(x_norm), y)
            loss_s.backward()
            opt_s.step()

    # recombine (CapsuleBD core)
    new_state = {}
    for k in global_model.state_dict():
        new_state[k] = (
            LAMBDA_POISON * model_poison.state_dict()[k]
            + LAMBDA_SHELL  * model_shell.state_dict()[k]
        )
    return new_state

# ======================================================
# HONEST LOCAL TRAINING
# ======================================================
def train_honest_local(model, loader):
    opt = make_optimizer(model)
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = NORMALIZER(x)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    return model.state_dict()

# ======================================================
# FEDAVG
# ======================================================
@torch.no_grad()
def fedavg(global_model, local_states):
    gsd = global_model.state_dict()
    new_state = {}

    for k in gsd:
        vals = [sd[k] for sd in local_states]
        v0 = vals[0]
        if torch.is_floating_point(v0):
            new_state[k] = torch.stack([v.float() for v in vals]).mean(0).type_as(gsd[k])
        else:
            new_state[k] = v0.clone()

    global_model.load_state_dict(new_state, strict=True)

# ======================================================
# METRICS
# ======================================================
@torch.no_grad()
def eval_acc(model):
    model.eval()
    c = t = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        p = model(x).argmax(1)
        c += (p == y).sum().item()
        t += y.size(0)
    return 100.0 * c / t

@torch.no_grad()
def eval_asr(model):
    model.eval()
    hit = tot = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        mask = (y != TARGET_LABEL)
        if mask.sum().item() == 0:
            continue
        xb = x[mask]
        xb = torch.stack([add_capsulebd_trigger(img) for img in xb])
        xb = NORMALIZER(xb)
        p = model(xb).argmax(1)
        hit += (p == TARGET_LABEL).sum().item()
        tot += p.size(0)
    return hit / max(tot, 1)

# ======================================================
# TRAIN LOOP
# ======================================================
if __name__ == "__main__":
    history = []
    os.makedirs("outputs", exist_ok=True)
    
    for r in trange(1, ROUNDS + 1):
        local_states = []
        t_infer_sec = 0.0
    
        for cid in range(NUM_CLIENTS):
            loader = client_loaders[cid]
    
            if cid in ATTACKER_IDX:
                st = train_capsulebd_local(global_model, loader)
            else:
                local = copy.deepcopy(global_model)
                st = train_honest_local(local, loader)
    
            local_states.append({k: v.detach().cpu() for k, v in st.items()})
    
            t0 = time.time()
            _ = eval_asr(global_model)
            t_infer_sec += time.time() - t0
    
        fedavg(global_model, local_states)
    
        acc = eval_acc(global_model)
        asr = eval_asr(global_model)
    
        print(f"R{r:02d} | ACC={acc:.2f}% | ASR={asr*100:.2f}%")
    
        history.append({
            "round": r,
            "acc": acc,
            "asr": asr * 100.0,
            "benign_drop_rate": 0.0,
            "server_queries": NUM_CLIENTS * REF_SIZE,
            "t_infer_sec": float(t_infer_sec),
            "t_entropy_mad_sec": 0.0
        })
    
    # ======================================================
    # SAVE
    # ======================================================
    with open(SAVE_PATH, "w") as f:
        json.dump({
            "dataset": "cifar10",
            "attack": "capsulebd",
            "defense": "fedavg",
            "rounds": ROUNDS,
            "results": history
        }, f, indent=2)
    
    print("\nFINAL")
    print(f"Clean ACC: {eval_acc(global_model):.2f}%")
    print(f"ASR: {eval_asr(global_model)*100:.2f}%")
    print(f"Saved to {SAVE_PATH}")
