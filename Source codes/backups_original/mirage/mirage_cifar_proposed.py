# ============================================================
# CIFAR-10 + MIRAGE Attack + BlackBoxGuard (Prediction-only)
# ============================================================

import os, time, json, random
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

NUM_CLIENTS = 4
ATTACKER_IDX = [2, 3]
ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 128
NUM_WORKERS = 2

LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 3000
REF_SIZE = 2000

# MIRAGE parameters
POISON_RATE = 0.2
TARGET_LABEL = 0
TRIGGER_SIZE = 4
TRIGGER_LR = 5e-2

# BlackBoxGuard
ALPHA_MAD = 2.0

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

test_loader = DataLoader(testset, batch_size=256, shuffle=False)
ref_subset = Subset(testset, list(range(REF_SIZE)))
ref_loader = DataLoader(ref_subset, batch_size=256, shuffle=False)

# ---------------- MIRAGE TRIGGER ----------------
IMAGENET_MEAN_T = torch.tensor(IMAGENET_MEAN).view(3,1,1).to(DEVICE)
IMAGENET_STD_T  = torch.tensor(IMAGENET_STD).view(3,1,1).to(DEVICE)

def init_trigger():
    trigger = 0.01 * torch.randn(3, TRIGGER_SIZE, TRIGGER_SIZE, device=DEVICE)
    trigger.requires_grad_(True)
    return trigger

def apply_trigger(x: torch.Tensor, trigger: torch.Tensor) -> torch.Tensor:
    """
    Works for:
      - single image: [3, H, W]
      - batch:        [B, 3, H, W]
    trigger: [3, ts, ts]
    """
    ts = trigger.shape[-1]

    if x.dim() == 3:
        # [3,H,W]
        x = x.clone()
        x[:, -ts:, -ts:] = trigger
        return x

    elif x.dim() == 4:
        # [B,3,H,W]
        x = x.clone()
        x[:, :, -ts:, -ts:] = trigger.unsqueeze(0)  # broadcast to batch
        return x

    else:
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")


# ---------------- MODEL ----------------
def make_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith("layer4") or n.startswith("fc"):
            p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

criterion_clean  = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion_attack = nn.CrossEntropyLoss()

def make_opt(model):
    return optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

# ---------------- MIRAGE LOCAL TRAIN ----------------
def train_mirage(global_model, loader, trigger):
    model = deepcopy(global_model)
    opt_m = make_opt(model)
    opt_t = optim.Adam([trigger], lr=TRIGGER_LR)

    model.train()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            mask = (torch.rand(len(x), device=DEVICE) < POISON_RATE) & (y != TARGET_LABEL)
            if mask.any():
                x_poison = apply_trigger(x[mask], trigger)
                y_poison = torch.full_like(y[mask], TARGET_LABEL)
                x = torch.cat([x, x_poison], dim=0)
                y = torch.cat([y, y_poison], dim=0)

            opt_m.zero_grad()
            opt_t.zero_grad()
            loss = criterion_attack(model(x), y)
            loss.backward()
            opt_m.step()
            opt_t.step()

    return model.state_dict()

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
    return 100 * c / t

@torch.no_grad()
def eval_asr(model, trigger):
    model.eval()
    hit = tot = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mask = torch.ones_like(y, dtype=torch.bool)
        if mask.sum().item() == 0:
            continue
        xt = apply_trigger(x[mask], trigger)
        pred = model(xt).argmax(1)
        hit += (pred == TARGET_LABEL).sum().item()
        tot += mask.sum().item()
    return 100.0 * hit / max(1, tot)

@torch.no_grad()
def eval_target_rate_clean(model):
    model.eval()
    hit = tot = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        mask = (y != TARGET_LABEL)
        if mask.sum().item() == 0:
            continue
        pred = model(x[mask]).argmax(1)
        hit += (pred == TARGET_LABEL).sum().item()
        tot += mask.sum().item()
    return 100.0 * hit / max(1, tot)



# ---------------- BLACKBOXGUARD ----------------
@torch.no_grad()
def entropy_on_ref_timed(model):
    model.eval()
    probs = []
    t0 = time.time()
    for x, _ in ref_loader:
        x = x.to(DEVICE)
        probs.append(F.softmax(model(x), dim=1).cpu())
    t_infer = time.time() - t0
    P = torch.cat(probs, dim=0)
    H = (-(P * (P + 1e-12).log()).sum(dim=1)).mean().item()
    return H, t_infer

def mad_filter(entropies):
    H = np.array(entropies)
    med = np.median(H)
    mad = np.median(np.abs(H - med)) + 1e-12
    keep = np.abs(H - med) <= ALPHA_MAD * mad
    if keep.sum() == 0:
        keep[:] = True
    return keep

def entropy_weights(H):
    H = np.array(H)
    w = (H.max() - H) / (H.max() - H.min() + 1e-6)
    return (w + 1e-3) / (w.sum() + 1e-12)

# ---------------- FEDERATED LOOP ----------------
global_model = make_model()
trigger = init_trigger()

metrics = {
    "round": [],
    "acc": [],
    "asr": [],
    "benign_drop_rate": [],
    "server_queries": [],
    "t_infer_sec": [],
    "t_entropy_mad_sec": [],
}

for r in range(1, ROUNDS + 1):
    print(f"\n=== Round {r}/{ROUNDS} ===")

    local_states, entropies = [], []
    t_infer_sum = 0.0

    for cid in range(NUM_CLIENTS):
        loader = DataLoader(
            client_datasets[cid],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        if cid in ATTACKER_IDX:
            sd = train_mirage(global_model, loader, trigger)
        else:
            m = deepcopy(global_model)
            opt = make_opt(m)
            for _ in range(LOCAL_EPOCHS):
                for x, y in loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion_clean(m(x), y)
                    loss.backward()
                    opt.step()
            sd = m.state_dict()

        tmp = make_model()
        tmp.load_state_dict(sd)
        H, tinf = entropy_on_ref_timed(tmp)

        local_states.append(sd)
        entropies.append(H)
        t_infer_sum += tinf

        t0_em = time.time()
        keep = mad_filter(entropies)
        t_entropy_mad = time.time() - t0_em


    benign_ids = [i for i in range(NUM_CLIENTS) if i not in ATTACKER_IDX]
    benign_drop = sum(not keep[i] for i in benign_ids) / max(1, len(benign_ids))

    kept_states = [sd for sd, k in zip(local_states, keep) if k]
    kept_H = [h for h, k in zip(entropies, keep) if k]

    if len(kept_states) == 0:
        kept_states = local_states
        kept_H = entropies

    w = entropy_weights(kept_H)

    new_sd = {}
    for k in kept_states[0]:
        new_sd[k] = sum(wi * sd[k] for wi, sd in zip(w, kept_states))
    global_model.load_state_dict(new_sd)

    acc = eval_acc(global_model)
    asr = eval_asr(global_model, trigger)

    print(f"ACC={acc:.2f}% | ASR={asr:.2f}% | BenignDrop={benign_drop:.2f}")

    metrics["round"].append(r)
    metrics["acc"].append(acc)
    metrics["asr"].append(asr)
    metrics["benign_drop_rate"].append(benign_drop)
    metrics["server_queries"].append(NUM_CLIENTS * REF_SIZE)
    metrics["t_infer_sec"].append(t_infer_sum)
    metrics["t_entropy_mad_sec"].append(t_entropy_mad)

# ---------------- SAVE ----------------
os.makedirs("outputs", exist_ok=True)
with open("outputs/cifar_mirage_blackboxguard.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved outputs/cifar_mirage_blackboxguard.json")
