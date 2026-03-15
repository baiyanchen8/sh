import os, time, json, random, copy
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ======================================================
# CONFIG
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET = "cifar10"
ATTACK = "mirage"
DEFENSE = "fedavg"  # no filtering, just aggregation

NUM_CLIENTS = 3
ATTACKER_IDX = 2

ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 128
NUM_WORKERS = 2

LR = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 3000

# MIRAGE attack parameters (from your mirage file style)
POISON_RATE = 0.20
TARGET_LABEL = 0
TRIGGER_SIZE = 4

# Metrics / logging
REF_SIZE = 2000
SAVE_PATH = "outputs/mirage_cifar_fedavg_metrics.json"

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
# MIRAGE TRIGGER (from your mirage file pattern)
# - init_trigger(): random patch (learnable in principle, but kept fixed here like your file)
# - apply_trigger(): overwrite bottom-right with trigger patch + clamp
# ======================================================
def init_trigger():
    # (3, TRIGGER_SIZE, TRIGGER_SIZE)
    return torch.randn(3, TRIGGER_SIZE, TRIGGER_SIZE, device=DEVICE)

def apply_trigger(x, trigger):
    """
    x: (B,3,224,224) normalized tensor batch
    trigger: (3,ts,ts) patch
    Overwrite bottom-right with the trigger patch, then clamp to [0,1] (same style as your mirage file).
    """
    xb = x.clone()
    xb[:, :, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = trigger
    return torch.clamp(xb, 0, 1)

trigger = init_trigger()

# ======================================================
# LOCAL TRAINING
# - Honest clients: normal training
# - Attackers: poison a fraction of each batch using apply_trigger + force target label
# ======================================================
def train_local(model, loader, attacker=False, trigger=None):
    model.train()
    opt = make_optimizer(model)

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = NORMALIZER(x)

            if attacker:
                mask = (torch.rand(x.size(0), device=DEVICE) < POISON_RATE)
                if mask.any():
                    x[mask] = apply_trigger(x[mask], trigger)
                    y[mask] = TARGET_LABEL

            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

    return model.state_dict()

# ======================================================
# FEDAVG
# ======================================================
@torch.no_grad()
def fedavg(global_model, local_states):
    """
    Average float tensors; copy integer/bool buffers (e.g., BatchNorm num_batches_tracked).
    """
    gsd = global_model.state_dict()
    new_state = {}

    for k in gsd.keys():
        vals = [sd[k] for sd in local_states]
        v0 = vals[0]
        if torch.is_floating_point(v0) or torch.is_complex(v0):
            stacked = torch.stack([v.float() for v in vals], dim=0)
            new_state[k] = stacked.mean(dim=0).type_as(gsd[k])
        else:
            new_state[k] = v0.clone()

    global_model.load_state_dict(new_state, strict=True)

# ======================================================
# METRICS
# ======================================================
@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)

@torch.no_grad()
def eval_asr(model, loader, trigger):
    """
    Targeted ASR: apply trigger to samples whose true label != TARGET_LABEL,
    count how many predictions become TARGET_LABEL.
    """
    model.eval()
    hit = tot = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        mask = (y != TARGET_LABEL)
        if mask.sum().item() == 0:
            continue

        xb = x[mask]
        xt = apply_trigger(xb, trigger)
        pred = model(xt).argmax(1)

        hit += (pred == TARGET_LABEL).sum().item()
        tot += pred.size(0)

    return hit / max(1, tot)

# ======================================================
# TRAIN LOOP + LOGGING
# ======================================================
if __name__ == "__main__":
    history = []
    os.makedirs("outputs", exist_ok=True)
    
    for r in trange(1, ROUNDS + 1):
        local_states = []
    
        # Metrics timers / queries (same schema as your other runs)
        t_infer_sec = 0.0
        server_queries = NUM_CLIENTS * REF_SIZE  # full participation
        benign_drop_rate = 0.0                   # no filtering in FedAvg
        t_entropy_mad_sec = 0.0                  # no entropy/MAD step
    
        for cid in range(NUM_CLIENTS):
            loader = client_loaders[cid]
            local_model = copy.deepcopy(global_model)
    
            attacker = (cid == ATTACKER_IDX)
            st = train_local(local_model, loader, attacker=attacker, trigger=trigger)
    
            local_states.append({k: v.detach().cpu() for k, v in st.items()})
    
            # "server inference time" on reference set (for overhead logging, like your defense runs)
            t0 = time.time()
            _ = eval_asr(local_model, ref_loader, trigger)
            t_infer_sec += time.time() - t0
    
        # aggregate
        fedavg(global_model, local_states)
    
        # evaluate global
        acc = eval_acc(global_model, test_loader)
        asr = eval_asr(global_model, test_loader, trigger)
    
        print(f"R{r:02d} | ACC={acc:.2f}% | ASR={asr*100:.2f}%")
    
        history.append({
            "round": r,
            "acc": float(acc),
            "asr": float(asr * 100.0),
            "benign_drop_rate": float(benign_drop_rate),
            "server_queries": int(server_queries),
            "t_infer_sec": float(t_infer_sec),
            "t_entropy_mad_sec": float(t_entropy_mad_sec),
        })
    
    # ======================================================
    # SAVE JSON
    # ======================================================
    with open(SAVE_PATH, "w") as f:
        json.dump({
            "dataset": DATASET,
            "attack": ATTACK,
            "defense": DEFENSE,
            "results": history
        }, f, indent=2)
    
    print("\nFINAL")
    print(f"Clean ACC: {eval_acc(global_model, test_loader):.2f}%")
    print(f"ASR: {eval_asr(global_model, test_loader, trigger)*100:.2f}%")
    print(f"Saved to {SAVE_PATH}")
