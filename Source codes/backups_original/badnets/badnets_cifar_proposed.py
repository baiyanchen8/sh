
import os, time, json, random, math
from copy import deepcopy
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

NUM_CLIENTS = 3                # 2 honest + 1 attacker (CIFAR only)
ATTACKER_IDX = 2               # attacker is client 2
ROUNDS = 20
BATCH_SIZE = 128
NUM_WORKERS = 2

LOCAL_EPOCHS = 3
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4

CIFAR_SUBSET_PER_CLIENT = 3000
REF_SIZE = 2000 
POISON_FRACTION = 0.005        # attacker poison fraction
PER_CLIENT_ASR_FLAG = 0.06     # absolute floor used inside robust flag (in ratio form)
TRIGGER_SIZE = 4
TRIGGER_VALUE = 1.0
TARGET_LABEL = 0



SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- TRANSFORMS ----------------
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform_cifar_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_cifar_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

NORMALIZER = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

# ---------------- DATA ----------------
train_cifar_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_cifar_train)
test_cifar = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar_test)

def make_client_subsets(dataset, per_client, n_groups):
    total_needed = per_client * n_groups
    total_available = len(dataset)
    take = min(total_needed, total_available - (total_available % n_groups))
    indices = list(range(take))
    subs = []
    per = take // n_groups
    for i in range(n_groups):
        start = i * per
        end = (i + 1) * per if i < n_groups - 1 else take
        subs.append(Subset(dataset, indices[start:end]))
    return subs

# 2 honest + 1 attacker CIFAR groups
client_datasets = make_client_subsets(train_cifar_full, CIFAR_SUBSET_PER_CLIENT, NUM_CLIENTS)

val_cifar_loader  = DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
test_cifar_loader = DataLoader(test_cifar, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
ref_indices = list(range(min(REF_SIZE, len(test_cifar))))
ref_subset = Subset(test_cifar, ref_indices)
ref_loader = DataLoader(ref_subset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)


# ---------------- BACKDOOR UTILITIES ----------------
def add_trigger(img):
    t = img.clone()
    _, h, w = t.shape
    y = h - TRIGGER_SIZE - 1
    x = w - TRIGGER_SIZE - 1
    t[:, y:y+TRIGGER_SIZE, x:x+TRIGGER_SIZE] = TRIGGER_VALUE
    return t

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, poison_fraction=POISON_FRACTION):
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

# ---------------- MODEL ----------------
def make_pretrained_resnet(num_classes=10):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

global_cifar = make_pretrained_resnet()

# freeze all except layer4 and fc
def freeze_except_layer4_fc(model):
    for _, p in model.named_parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True

freeze_except_layer4_fc(global_cifar)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

def make_optimizer_for(model, lr=LR_HEAD):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)

def train_local(model, loader, epochs=1, lr=LR_HEAD):
    model.train()
    opt = make_optimizer_for(model, lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def eval_asr(model, loader):
    model.eval()
    hit = tot = 0
    for x, _ in loader:
        xt = torch.stack([add_trigger(img) for img in x]).to(DEVICE)
        preds = model(xt).argmax(1)
        hit += (preds == TARGET_LABEL).sum().item()
        tot += preds.size(0)
    return 100.0 * hit / tot

# ---------------- STATE / AGG HELPERS ----------------
def state_to_vec(sd):
    out = []
    shapes = []
    for k, v in sd.items():
        arr = v.cpu().numpy().reshape(-1)
        shapes.append((k, v.shape))
        out.append(arr)
    return np.concatenate(out), shapes

def vec_to_state(vec, shapes, device=DEVICE):
    st = {}
    idx = 0
    for k, s in shapes:
        size = int(np.prod(s))
        arr = vec[idx:idx+size].reshape(s)
        st[k] = torch.tensor(arr, dtype=torch.float32).to(device)
        idx += size
    return st

def dict_minus(a, b):
    return {k: (a[k].float() - b[k].float()) for k in a}

def dict_l2_norm(d):
    return math.sqrt(sum((v.double()**2).sum().item() for v in d.values()))

def clip_dict_norm(d, thr):
    n = dict_l2_norm(d)
    if n == 0 or n <= thr:
        return d
    s = thr / (n + 1e-12)
    return {k: v * s for k, v in d.items()}

def coord_median(vecs):
    return np.median(np.stack(vecs), axis=0)

# ---------------- CLIENT LOADERS ----------------
client_loaders = []
for i, ds in enumerate(client_datasets):
    if i == ATTACKER_IDX:
        loader = DataLoader(PoisonedDataset(ds, POISON_FRACTION), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    else:
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    client_loaders.append(loader)

# ---------------- FED TRAIN ----------------
metrics = {"round": [], "acc": [], "asr": [],"benign_drop_rate": [], "server_queries": [], "t_infer_sec": [], "t_entropy_mad_sec": [],}
start_time = time.time()

def server_finetune(global_model, clean_loader, steps=5, lr_ft=LR_HEAD):
    global_model.train()
    params = [p for n, p in global_model.named_parameters() if p.requires_grad]
    opt = optim.SGD(params, lr=lr_ft, momentum=0.9, weight_decay=WEIGHT_DECAY)
    it = 0
    for x, y in clean_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = global_model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        it += 1
        if it >= steps:
            break

# CIFAR base server reference (recomputed each round after global updates is fine)
for r in range(1, ROUNDS + 1):
    t0 = time.time()
    t_infer_sum = 0.0
    t_entropy_mad = 0.0
    benign_drop_rate = 0.0
    server_queries = 0
    print(f"\n=== Round {r}/{ROUNDS} ===")

    sampled = list(range(NUM_CLIENTS))  # full participation
    local_states = {}
    client_asrs = {}
    norms = []

    # fixed CIFAR reference for norms this round
    base_server = {k: v.detach().cpu().clone() for k, v in global_cifar.state_dict().items()}

    for i in sampled:
        base = deepcopy(global_cifar).to(DEVICE)
        train_local(base, client_loaders[i], epochs=LOCAL_EPOCHS, lr=LR_HEAD)

        st = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        local_states[i] = st

        t_inf0 = time.time()
        client_asrs[i] = eval_asr(base, ref_loader)   # inference on V
        t_inf1 = time.time()
        t_infer_sum += (t_inf1 - t_inf0)


        n = dict_l2_norm(dict_minus(st, base_server))
        norms.append(n)
        print(f"Client {i}: ASR={client_asrs[i]:.2f}% norm={n:.2f}")

    med_norm = float(np.median(norms)) if len(norms) else 1.0
    clip_thr = med_norm * 1.0
    t_em0 = time.time()
    # robust ASR-based flagging (median + MAD) with absolute floor
    asr_list = np.array([client_asrs[i] for i in sampled], dtype=np.float64)
    asr_med = np.median(asr_list)
    asr_mad = np.median(np.abs(asr_list - asr_med)) + 1e-9

    K = 2.0
    ABS_FLOOR = PER_CLIENT_ASR_FLAG * 100.0  # percent
    asr_flags = (asr_list > (asr_med + K * asr_mad)) & (asr_list > ABS_FLOOR)
    flagged = set(i for idx, i in enumerate(sampled) if asr_flags[idx])

    if len(flagged) == len(sampled):
        flagged = set()
        print("All flagged by ASR rule — fallback: no flagging this round")
    t_em1 = time.time()
    t_entropy_mad = (t_em1 - t_em0)
    print("Flagged:", flagged)
    benign_selected = [i for i in sampled if i != ATTACKER_IDX]
    benign_flagged = [i for i in benign_selected if i in flagged]
    benign_drop_rate = len(benign_flagged) / max(1, len(benign_selected))

    server_queries = len(sampled) * REF_SIZE

    # CIFAR aggregation (coordinate-wise median of clipped deltas)
    vecs = []
    shapes_ref = None
    for i in sampled:
        if i in flagged:
            continue
        d = dict_minus(local_states[i], base_server)
        d = clip_dict_norm(d, clip_thr)
        vec, shapes = state_to_vec(d)
        vecs.append(vec)
        if shapes_ref is None:
            shapes_ref = shapes

    if len(vecs) > 0:
        agg = coord_median(vecs)
        agg_d = vec_to_state(agg, shapes_ref)
        new_state = {}
        for k, v in global_cifar.state_dict().items():
            new_state[k] = (v.detach().cpu().float() + agg_d[k].detach().cpu().float()).to(DEVICE)
        global_cifar.load_state_dict(new_state)

    # server fine-tune on clean CIFAR test batches (as you had)
    server_finetune(
        global_cifar,
        DataLoader(test_cifar, batch_size=256, shuffle=True, num_workers=NUM_WORKERS),
        steps=5,
        lr_ft=LR_HEAD
    )

    acc = eval_acc(global_cifar, test_cifar_loader)
    asr = eval_asr(global_cifar, test_cifar_loader)

    elapsed = time.time() - t0
    print(f"Round {r} done in {elapsed:.1f}s | ACC={acc:.2f}% ASR={asr:.2f}%")
    metrics = []

    metrics.append({
    "round": r,
    "acc": acc,
    "asr": asr,
    "benign_drop_rate": benign_drop_rate,
    "server_queries": server_queries,
    "t_infer_sec": t_infer_sum,
    "t_entropy_mad_sec": t_entropy_mad
})

# ---------------- SAVE ----------------
os.makedirs("outputs", exist_ok=True)
torch.save(global_cifar.state_dict(), "outputs/model_cifar.pth")
with open("outputs/metrics_cifar.json", "w") as f:
    json.dump(metrics, f, indent=2)

total_min = (time.time() - start_time) / 60.0
print("\nSaved outputs/model_cifar.pth, outputs/metrics_cifar.json")
print(f"Approx total runtime: {total_min:.2f} minutes")
