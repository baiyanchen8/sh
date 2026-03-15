# CIFAR-10 + Mirage Backdoor + RFOut-1d Aggregation

import os, time, json, random
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

NUM_CLIENTS = 4
ATTACKER_IDX = [2, 3]      
ROUNDS = 20
LOCAL_EPOCHS = 1

BATCH_SIZE = 128
NUM_WORKERS = 2

LR_HEAD = 3e-4
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 1500
REF_SIZE = 2000

# Mirage parameters
POISON_RATE = 0.2
TARGET_LABEL = 0
TRIGGER_SIZE = 4

# RFOut
RFOUT_DELTA = 3.0
RFOUT_EPS = 1e-12
RFOUT_UNBIASED_STD = False

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
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

NORMALIZER = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


# ---------------- DATA ----------------
trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

def split_clients(dataset, per_client, n):
    idx = list(range(per_client * n))
    return [Subset(dataset, idx[i*per_client:(i+1)*per_client]) for i in range(n)]

client_sets = split_clients(trainset, CIFAR_SUBSET_PER_CLIENT, NUM_CLIENTS)

test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# ---------------- MODEL ----------------
def make_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("layer4") or n.startswith("fc")
    return model.to(DEVICE)

global_model = make_model()
criterion = nn.CrossEntropyLoss()

def make_opt(m):
    return optim.SGD(
        [p for p in m.parameters() if p.requires_grad],
        lr=LR_HEAD, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

# ---------------- MIRAGE TRIGGER ----------------
def init_trigger():
    # keep trigger in normalized-value scale
    return torch.randn(3, TRIGGER_SIZE, TRIGGER_SIZE, device=DEVICE) * 0.5
trigger = init_trigger()

def apply_trigger(x):
    x = x.clone()
    x[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = trigger
    return x  # NO clamp


# ---------------- LOCAL TRAIN ----------------
def local_train(base_model, loader, attacker=False):
    m = deepcopy(base_model)
    opt = make_opt(m)
    m.train()

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = NORMALIZER(x)

            if attacker:
                mask = torch.rand(len(x), device=DEVICE) < POISON_RATE
                if mask.any():
                    x[mask] = apply_trigger(x[mask])
                    y[mask] = TARGET_LABEL

            opt.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            opt.step()

    upd = {}
    for k in base_model.state_dict():
        upd[k] = (m.state_dict()[k] - base_model.state_dict()[k]).detach()
    return upd

# ---------------- RFOUT AGGREGATION ----------------
@torch.no_grad()
def rfout1d_aggregate_updates(client_updates):
    n = len(client_updates)
    keys = list(client_updates[0].keys())
    agg = {}

    for k in keys:
        stacked = torch.stack([u[k] for u in client_updates], dim=0).float()
        mu = stacked.mean(dim=0)
        sigma = stacked.std(dim=0)
        mask = (stacked - mu).abs() >= RFOUT_DELTA * (sigma + RFOUT_EPS)
        stacked = torch.where(mask, mu.unsqueeze(0), stacked)
        agg[k] = stacked.mean(dim=0)

    return agg

# ---------------- EVAL ----------------
@torch.no_grad()
def eval_acc(m):
    m.eval()
    c = t = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        c += (m(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return 100 * c / t

@torch.no_grad()
def eval_asr(m):
    m.eval()
    hit = tot = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        mask = (y != TARGET_LABEL)
        if mask.sum() == 0:
            continue
        xt = apply_trigger(x[mask])
        hit += (m(xt).argmax(1) == TARGET_LABEL).sum().item()
        tot += mask.sum().item()
    return 100 * hit / max(1, tot)


# ---------------- TRAIN ----------------
metrics = {k: [] for k in [
    "round", "acc", "asr"
]}

if __name__ == "__main__":
    for r in range(1, ROUNDS + 1):
        print(f"\n=== Round {r}/{ROUNDS} ===")
        base = deepcopy(global_model)
    
        client_updates = []
    
        for i, ds in enumerate(client_sets):
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
            upd = local_train(base, loader, attacker=(i in ATTACKER_IDX))
            client_updates.append(upd)
    
        # RFOut aggregation
        t0 = time.time()
        agg_update = rfout1d_aggregate_updates(client_updates)
        t_rfout = time.time() - t0
    
        new_state = {}
        for k in global_model.state_dict():
            new_state[k] = (global_model.state_dict()[k] + agg_update[k]).detach()
        global_model.load_state_dict(new_state)
    
        acc = eval_acc(global_model)
        asr = eval_asr(global_model)
    
        print(f"ACC={acc:.2f}% ASR={asr:.2f}%")
    
        metrics["round"].append(r)
        metrics["acc"].append(acc)
        metrics["asr"].append(asr)
    
    # ---------------- SAVE ----------------
    os.makedirs("outputs", exist_ok=True)
    torch.save(global_model.state_dict(), "outputs/rfout_mirage_cifar.pth")
    json.dump(metrics, open("outputs/rfout_mirage_cifar.json", "w"), indent=2)
    
    print("\nSaved Mirage + RFOut results successfully.")
