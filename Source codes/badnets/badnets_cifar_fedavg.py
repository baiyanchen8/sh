import os, time, json, random, copy
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset


# CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_CLIENTS = 3
ATTACKER_IDX = 2

ROUNDS = 20
LOCAL_EPOCHS_BENIGN = 3
LOCAL_EPOCHS_ATTACKER = 8

BATCH_SIZE = 128
NUM_WORKERS = 2

LR = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 3000

POISON_FRACTION = 0.20  # strong for validation baseline
TARGET_LABEL = 0
TRIGGER_SIZE = 4
TRIGGER_VALUE = 1.0

REF_SIZE = 2000
SAVE_PATH = "outputs/badnets_cifar_fedavg_metrics.json"

# DATA
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Benign (keep your original augmentation)
transform_train_clean = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Attacker (NO spatial augmentation -> preserves trigger location)
transform_train_attack = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

NORMALIZER = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

trainset_clean  = datasets.CIFAR10("./data", train=True,  download=True, transform=transform_train_clean)
trainset_attack = datasets.CIFAR10("./data", train=True,  download=True, transform=transform_train_attack)
testset         = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

def make_client_subsets(train_clean, train_attack, per_client, n_clients, attacker_idx):
    indices = list(range(min(len(train_clean), per_client * n_clients)))
    subsets = []
    for cid in range(n_clients):
        base_ds = train_attack if cid == attacker_idx else train_clean
        start = cid * per_client
        end = (cid + 1) * per_client
        subsets.append(Subset(base_ds, indices[start:end]))
    return subsets

client_datasets = make_client_subsets(
    trainset_clean,
    trainset_attack,
    CIFAR_SUBSET_PER_CLIENT,
    NUM_CLIENTS,
    ATTACKER_IDX
)

test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
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


# BADNETS TRIGGER 
GLOBAL_TRIGGER = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
GLOBAL_TRIGGER[:, -TRIGGER_SIZE:, -TRIGGER_SIZE:] = TRIGGER_VALUE
GLOBAL_TRIGGER = GLOBAL_TRIGGER  # (3,224,224) on CPU; move to device when used

def add_trigger(img):
    """
    img is un-normalized (0-1).
    ADD the trigger and clamp
    """
    x = img.clone()
    trig = GLOBAL_TRIGGER.to(x.device)
    x = torch.clamp(x + trig, 0, 1)
    return x

class PoisonedDataset(Dataset):
    def __init__(self, subset, poison_fraction):
        self.subset = subset
        self.n = len(self.subset)
        self.num_poison = max(1, int(self.n * poison_fraction))
        self.poison_idx = set(random.sample(range(self.n), self.num_poison))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if idx in self.poison_idx:
            x = add_trigger(x)
            y = TARGET_LABEL
        return x, y

client_loaders = []
for cid, ds in enumerate(client_datasets):
    if cid == ATTACKER_IDX:
        ds = PoisonedDataset(ds, POISON_FRACTION)
    client_loaders.append(
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    )

# MODEL (ResNet-18, train all)
model_global = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_global.fc = nn.Linear(model_global.fc.in_features, 10)
model_global = model_global.to(DEVICE)

criterion = nn.CrossEntropyLoss()

def make_optimizer(model):
    return optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)

def train_local(model, loader, epochs):
    model.train()
    opt = make_optimizer(model)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = NORMALIZER(x)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()


#FEDAVG
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


# METRICS
@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = NORMALIZER(x)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def eval_asr(model, loader):
    model.eval()
    hit, tot = 0, 0

   
    trig = GLOBAL_TRIGGER.to(DEVICE).unsqueeze(0)  # (1,3,224,224)

    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        mask = (y != TARGET_LABEL)
        if mask.sum().item() == 0:
            continue

       
        xb = x[mask]  # (B',3,224,224)

        # ADD trigger and clamp (same as training poisoning)
        xt = torch.clamp(xb + trig, 0, 1)
        xt = NORMALIZER(xt)

        pred = model(xt).argmax(1)
        hit += (pred == TARGET_LABEL).sum().item()
        tot += pred.size(0)

    return hit / max(1, tot)



# TRAIN LOOP + METRICS LOGGING (no defense => benign_drop_rate=0, t_entropy_mad_sec=0)
if __name__ == "__main__":
    history = []
    os.makedirs("outputs", exist_ok=True)
    
    for r in trange(1, ROUNDS + 1):
        local_states = []
    
        # Metrics timers / queries 
        #t_infer_sec = 0.0
        #server_queries = NUM_CLIENTS * REF_SIZE  # full participation here
        #benign_drop_rate = 0.0
        #t_entropy_mad_sec = 0.0
    
        # train all clients (full participation)
        for cid in range(NUM_CLIENTS):
            local_model = copy.deepcopy(model_global)
            epochs = LOCAL_EPOCHS_ATTACKER if cid == ATTACKER_IDX else LOCAL_EPOCHS_BENIGN
            train_local(local_model, client_loaders[cid], epochs=epochs)
    
            # Save local state for FedAvg
            st = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
            local_states.append(st)
    
            # "server inference time" on reference set 
            #t0 = time.time()
            #asr = eval_asr(local_model, ref_loader)
            #t_infer_sec += (time.time() - t0)
    
        # aggregate
        fedavg(model_global, local_states)
    
        # evaluate global
        acc = eval_acc(model_global, test_loader)
        asr = eval_asr(model_global, test_loader)
    
        print(f"Round {r:02d} | Clean ACC = {acc:.2f}% | ASR = {asr*100:.2f}%")
    
        history.append({
            "round": r,
            "acc": acc,
            "asr": asr * 100.0,
            #"benign_drop_rate": benign_drop_rate,
           # "server_queries": int(server_queries),
            #"t_infer_sec": float(t_infer_sec),
            #"t_entropy_mad_sec": float(t_entropy_mad_sec),
        })
    
    # SAVE RESULTS 
    with open(SAVE_PATH, "w") as f:
        json.dump({
            "dataset": "cifar10",
            "attack": "badnets",
            "defense": "fedavg",
            "rounds": ROUNDS,
            "results": history
        }, f, indent=2)
    
    print("\nFINAL")
    print(f"Clean Accuracy: {eval_acc(model_global, test_loader):.2f}%")
    print(f"ASR: {eval_asr(model_global, test_loader)*100:.2f}%")
    print(f"Saved results to {SAVE_PATH}")
