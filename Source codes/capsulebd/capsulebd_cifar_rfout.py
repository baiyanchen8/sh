# CIFAR-10 + CapsuleBD Backdoor + RFOut-1d Aggregation

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
ATTACKER_EPOCHS = 2

BATCH_SIZE = 128
NUM_WORKERS = 2

LR_HEAD = 3e-4
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 1500
REF_SIZE = 2000

# CapsuleBD parameters (from your file)
POISON_RATE = 0.2
TARGET_LABEL = 0
TRIGGER_SIZE = 4
TRIGGER_ALPHA = 0.25
LAMBDA_POISON = 0.7
LAMBDA_SHELL = 0.3

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

ref_subset = Subset(testset, list(range(REF_SIZE)))
ref_loader = DataLoader(ref_subset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# ---------------- MODEL ----------------
def make_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith("layer4") or n.startswith("fc")
    return model.to(DEVICE)

global_model = make_model()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def make_opt(m):
    return optim.SGD([p for p in m.parameters() if p.requires_grad],
                     lr=LR_HEAD, momentum=0.9, weight_decay=WEIGHT_DECAY)
# ---------------- CAPSULEBD TRIGGER ----------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

WHITE_NORM = torch.tensor([(1-m)/s for m,s in zip(IMAGENET_MEAN, IMAGENET_STD)]).view(3,1,1)

def add_trigger(x):
    t = x.clone()
    _, h, w = t.shape
    y0 = h - TRIGGER_SIZE
    x0 = w - TRIGGER_SIZE
    t[:, y0:y0+TRIGGER_SIZE, x0:x0+TRIGGER_SIZE] = WHITE_NORM.to(t.device)
    return t


# ---------------- CAPSULEBD LOCAL TRAIN ----------------
def train_capsulebd(base_model, loader):
    model_poison = deepcopy(base_model)
    model_shell  = deepcopy(base_model)

    opt_p = make_opt(model_poison)
    opt_s = make_opt(model_shell)

    model_poison.train()
    model_shell.train()

    for _ in range(ATTACKER_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = NORMALIZER(x)

            # ---- poison branch ----
            xp = x.clone()
            yp = y.clone()
            for i in range(len(xp)):
                if random.random() < POISON_RATE:
                    xp[i] = add_trigger(xp[i])
                    yp[i] = TARGET_LABEL

            opt_p.zero_grad()
            loss_p = criterion(model_poison(xp), yp)
            loss_p.backward()
            opt_p.step()

            # ---- shell branch ----
            opt_s.zero_grad()
            loss_s = criterion(model_shell(x), y)
            loss_s.backward()
            opt_s.step()

    # recombine (CapsuleBD core)
    upd = {}
    for k in base_model.state_dict():
        upd[k] = (
            LAMBDA_POISON * (model_poison.state_dict()[k] - base_model.state_dict()[k]) +
            LAMBDA_SHELL  * (model_shell.state_dict()[k]  - base_model.state_dict()[k])
        ).detach()
    return upd

# ---------------- HONEST UPDATE ----------------
def honest_update(base_model, loader):
    m = deepcopy(base_model)
    opt = make_opt(m)
    m.train()

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = NORMALIZER(x)
            opt.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            opt.step()

    upd = {}
    for k in base_model.state_dict():
        upd[k] = (m.state_dict()[k] - base_model.state_dict()[k]).detach()
    return upd



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
    for x, _ in test_loader:
        xt = torch.stack([add_trigger(img) for img in x]).to(DEVICE)
        xt = NORMALIZER(xt)
        hit += (m(xt).argmax(1) == TARGET_LABEL).sum().item()
        tot += xt.size(0)
    return 100 * hit / tot

# ---------------- TRAINING UTILS ----------------
def model_update(base_model, loader, steps=1):
    m = deepcopy(base_model)
    opt = make_opt(m)
    m.train()
    it = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        opt.zero_grad()
        loss = criterion(m(x), y)
        loss.backward()
        opt.step()

    # return update (delta)
    base_sd = base_model.state_dict()
    new_sd = m.state_dict()
    upd = {}
    for k in base_sd:
        upd[k] = (new_sd[k] - base_sd[k]).detach()
    return upd

# ---------------- RFOUT-1D AGGREGATION  ----------------
@torch.no_grad()
def rfout1d_aggregate_updates(client_updates, delta=3.0, unbiased_std=False, eps=1e-12):
    n = len(client_updates)
    keys = list(client_updates[0].keys())

    agg = {}
    client_flag_any = torch.zeros(n, dtype=torch.bool)

    for k in keys:
        stacked = torch.stack([u[k].detach() for u in client_updates], dim=0)

        stats_dtype = stacked.dtype if torch.is_floating_point(stacked) else torch.float32
        x = stacked.to(dtype=stats_dtype)

        mu = x.mean(dim=0)
        sigma = x.std(dim=0, unbiased=unbiased_std)

        thr = delta * (sigma + eps)
        mask = (x - mu).abs() >= thr

        # proxy metric only (NOT part of RFOut algorithm)
        client_flag_any |= mask.reshape(mask.size(0), -1).any(dim=1).cpu()

        x_f = torch.where(mask, mu.unsqueeze(0).expand_as(x), x)
        agg[k] = x_f.mean(dim=0).to(stacked.dtype)

    return agg, client_flag_any.tolist()




# ---------------- TRAIN ----------------
metrics = {k: [] for k in [
    "round", "acc", "asr",    
]}

if __name__ == "__main__":
    for r in range(1, ROUNDS + 1):
        print(f"\n=== Round {r}/{ROUNDS} ===")
        base = deepcopy(global_model)
    
        client_updates = []
        t_inf = 0.0
    
        for i, ds in enumerate(client_sets):
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
            if i in ATTACKER_IDX:
                upd = train_capsulebd(base, loader)   # CapsuleBD attacker
            else:
                upd = honest_update(base, loader)     # honest client
    
            client_updates.append(upd)
    
        # keep your timing logic if you want
        ti = time.time()
        _ = eval_asr(deepcopy(base))
        t_inf += time.time() - ti
    
        # RFOut aggregation timing (logged under t_entropy_mad_sec as requested)
        t_agg0 = time.time()
        agg_update, client_flag_any = rfout1d_aggregate_updates(
            client_updates,
            delta=RFOUT_DELTA,
            unbiased_std=RFOUT_UNBIASED_STD,
            eps=RFOUT_EPS
        )
        t_rfout_sec = time.time() - t_agg0
    
        # apply update to global
        new_state = {}
        gsd = global_model.state_dict()
        for k in gsd:
            new_state[k] = (gsd[k] + agg_update[k]).detach()
        global_model.load_state_dict(new_state)
    
        acc = eval_acc(global_model)
        asr = eval_asr(global_model)
    
        # "Benign drop" proxy for RFOut: benign clients that were flagged
        n_flags = len(client_flag_any)
        benign_idx = [i for i in range(NUM_CLIENTS) if i not in ATTACKER_IDX]
        benign_flagged = sum(1 for i in benign_idx if client_flag_any[i])
        benign_drop = benign_flagged / max(1, len(benign_idx))
    
        print(f"ACC={acc:.2f}% ASR={asr:.2f}%")
    
        metrics["round"].append(r)
        metrics["acc"].append(acc)
        metrics["asr"].append(asr)
    # ---------------- SAVE ----------------
    os.makedirs("outputs", exist_ok=True)
    torch.save(global_model.state_dict(), "outputs/rfout_capsulebd_cifar.pth")
    json.dump(metrics, open("outputs/rfout_capsulebd_cifar.json", "w"), indent=2)
    
    print("\nSaved CapsuleBD + RFOut results successfully.")
