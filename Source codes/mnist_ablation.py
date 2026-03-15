import math, random, copy, time
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------- Transforms -----------------------
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test  = transforms.Compose([transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- FL hyperparams ----------------------
NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 10
GLOBAL_ROUNDS = 200
LOCAL_EPOCHS = 2
BATCH_SIZE = 64

# Model & optimization
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Defense (BlackBoxGuard)
REF_SIZE = 2000
ALPHA = 2.0
EPS = 1e-6

# Attack (BadNets-style)
NUM_MALICIOUS = 5
POISON_RATE = 0.2
TRIGGER_VALUE = 1.0
TRIGGER_SIZE = 3
TARGET_LABEL = 0

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------- Model ---------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# -------------------- Data ---------------------
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# Make a balanced server-side reference set (disjoint from client shards)
all_indices = list(range(len(trainset)))
random.shuffle(all_indices)

ref_indices = []
class_counts = defaultdict(int)
target_per_class = REF_SIZE // 10
i = 0
while len(ref_indices) < REF_SIZE and i < len(all_indices):
    idx = all_indices[i]
    _, label = trainset[idx]
    if class_counts[label] < target_per_class:
        ref_indices.append(idx)
        class_counts[label] += 1
    i += 1

ref_set = set(ref_indices)
client_indices = [idx for idx in all_indices if idx not in ref_set]
random.shuffle(client_indices)

shard_size = len(client_indices) // NUM_CLIENTS
clients_data = []
for ci in range(NUM_CLIENTS):
    start = ci * shard_size
    end = start + shard_size if ci < NUM_CLIENTS - 1 else len(client_indices)
    clients_data.append(client_indices[start:end])

ref_dataset = Subset(trainset, ref_indices)
ref_loader  = DataLoader(ref_dataset, batch_size=128, shuffle=False, num_workers=2)


malicious_ids = set(random.sample(range(NUM_CLIENTS), NUM_MALICIOUS))
print("Malicious client ids:", malicious_ids)
print(f"Attack config: poison_rate={POISON_RATE}, attackers={NUM_MALICIOUS}, target={TARGET_LABEL}, trigger={TRIGGER_SIZE}x{TRIGGER_SIZE}")
print(f"Defense config: REF_SIZE={REF_SIZE}, ALPHA={ALPHA}")

# --------------------- Helpers ----------------------
def add_trigger(img_tensor, size=TRIGGER_SIZE, value=TRIGGER_VALUE):
    x = img_tensor.clone()          # (1,28,28)
    x[..., -size:, -size:] = value
    return x

def make_client_dataset(indices, attacker=False):
    imgs, labels = [], []
    for idx in indices:
        img, label = trainset[idx]  # img: (1,28,28)
        if attacker and random.random() < POISON_RATE:
            img = add_trigger(img)
            label = TARGET_LABEL
        imgs.append(img)
        labels.append(label)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long)

def local_train(model, data_tensor, label_tensor, epochs=LOCAL_EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())

@torch.no_grad()
def get_prediction_probs(model_state_dict):
    m = SimpleCNN().to(device)
    m.load_state_dict(model_state_dict)
    m.eval()
    probs = []
    for xb, _ in ref_loader:
        xb = xb.to(device)
        logits = m(xb)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)

def mean_shannon_entropy(probs):
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    return float(ent.mean())

def weighted_aggregate(state_dicts, weights):
    agg = {}
    total_w = float(sum(weights)) + EPS
    for k in state_dicts[0].keys():
        agg[k] = sum(state_dicts[i][k].float() * (weights[i] / total_w) for i in range(len(state_dicts)))
    return agg

def plain_aggregate(state_dicts):
    return weighted_aggregate(state_dicts, [1.0] * len(state_dicts))

# Triggered test subset for ASR
triggered_images = []
for xb, _ in test_loader:
    for i in range(xb.size(0)):
        img = xb[i].clone()             # (1,28,28)
        img = add_trigger(img)          # (1,28,28)
        triggered_images.append(img.unsqueeze(0))  # (1,1,28,28)
    if len(triggered_images) >= 1000:
        break
triggered_images = torch.cat(triggered_images[:1000], dim=0)

@torch.no_grad()
def compute_asr(state_dict):
    m = SimpleCNN().to(device)
    m.load_state_dict(state_dict)
    m.eval()
    correct_attack = 0
    bsz = 128
    for i in range(0, len(triggered_images), bsz):
        xb = triggered_images[i:i+bsz].to(device)
        out = m(xb)
        preds = out.argmax(dim=1)
        correct_attack += (preds == TARGET_LABEL).sum().item()
    return correct_attack / len(triggered_images)

@torch.no_grad()
def evaluate_clean_acc(state_dict):
    m = SimpleCNN().to(device)
    m.load_state_dict(state_dict)
    m.eval()
    correct, total = 0, 0
    for xb, yb in test_loader:
        xb = xb.to(device); yb = yb.to(device)
        out = m(xb)
        pred = out.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total

# -------------------- One run function ---------------------
def run_fl(setting_name, use_filtering, use_weighting):
    global_state = copy.deepcopy(SimpleCNN().to(device).state_dict())

    history = {
        'round': [], 'clean_acc': [], 'asr': [],
        'benign_drop_rate': [],
        'server_queries': [], 't_infer_sec': [], 't_entropy_mad_sec': []
    }

    for rnd in trange(1, GLOBAL_ROUNDS + 1, desc=setting_name):
        selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)

        client_states = []
        entropies = []

        # overhead counts
        server_queries = CLIENTS_PER_ROUND * REF_SIZE
        t_infer = 0.0

        # train clients + (if needed) compute entropy via server-side inference
        for cid in selected:
            attacker = cid in malicious_ids
            data_tensor, label_tensor = make_client_dataset(clients_data[cid], attacker=attacker)

            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_state)
            updated_state = local_train(local_model, data_tensor, label_tensor)
            client_states.append(updated_state)

            if use_filtering or use_weighting:
                t0 = time.time()
                probs = get_prediction_probs(updated_state)
                t_infer += (time.time() - t0)
                entropies.append(mean_shannon_entropy(probs))

        # if no defense at all, just aggregate
        benign_drop_rate = 0.0
        t_entmad = 0.0

        if not (use_filtering or use_weighting):
            global_state = plain_aggregate(client_states)
        else:
            # compute robust stats
            t1 = time.time()
            ent_arr = np.array(entropies, dtype=np.float64)
            med = float(np.median(ent_arr))
            mad = float(np.median(np.abs(ent_arr - med))) + 1e-12

            # filtering
            if use_filtering:
                flags = np.abs(ent_arr - med) > ALPHA * mad
                retained_indices = [i for i, f in enumerate(flags) if not f]
                if len(retained_indices) == 0:
                    retained_indices = list(range(len(client_states)))
                    flags = np.array([False] * len(client_states), dtype=bool)
            else:
                flags = np.array([False] * len(client_states), dtype=bool)
                retained_indices = list(range(len(client_states)))

            # benign-drop
            benign_filtered = 0
            benign_selected = 0
            for i, cid in enumerate(selected):
                if cid not in malicious_ids:
                    benign_selected += 1
                    if flags[i]:
                        benign_filtered += 1
            benign_drop_rate = (benign_filtered / benign_selected) if benign_selected > 0 else 0.0

            retained_states = [client_states[i] for i in retained_indices]

            # weighting
            if use_weighting:
                retained_ents = ent_arr[retained_indices]
                Hmax = float(retained_ents.max())
                Hmin = float(retained_ents.min())
                weights = []
                for idx in retained_indices:
                    Hi = float(ent_arr[idx])
                    w = (Hmax - Hi) / (Hmax - Hmin + EPS)
                    weights.append(w + 1e-3)
                global_state = weighted_aggregate(retained_states, weights)
            else:
                global_state = plain_aggregate(retained_states)

            t_entmad = time.time() - t1

        # metrics
        clean_acc = evaluate_clean_acc(global_state)
        asr = compute_asr(global_state)

        history['round'].append(rnd)
        history['clean_acc'].append(clean_acc)
        history['asr'].append(asr)
        history['benign_drop_rate'].append(benign_drop_rate)
        history['server_queries'].append(server_queries if (use_filtering or use_weighting) else 0)
        history['t_infer_sec'].append(t_infer if (use_filtering or use_weighting) else 0.0)
        history['t_entropy_mad_sec'].append(t_entmad if (use_filtering or use_weighting) else 0.0)

    # summarize
    summary = {
        'Setting': setting_name,
        'Final_CA': history['clean_acc'][-1] * 100,
        'Final_ASR': history['asr'][-1] * 100,
        'Avg_Benign_Drop': float(np.mean(history['benign_drop_rate'])) * 100,
        'Avg_Queries': float(np.mean(history['server_queries'])),
        'Avg_Infer_s': float(np.mean(history['t_infer_sec'])),
        'Avg_EntMAD_s': float(np.mean(history['t_entropy_mad_sec'])),
        'History': history
    }
    return summary

# --------------------- Run all ablations in one script ----------------------
ablations = [
    ("Attack only no defense", False, False),
    ("Filtering only", True, False),
    ("Weighting only", False, True),
    ("Full BlackBoxGuard", True, True),
]

summaries = []
for name, f_on, w_on in ablations:
    # keep runs comparable: reset RNG per setting (optional)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    summaries.append(run_fl(name, f_on, w_on))

# --------------------- Print summary table ---------------------
print("\n===== Ablation Summary (MNIST) =====")
for s in summaries:
    print(
        f"{s['Setting']} | "
        f"CA {s['Final_CA']:.2f}% | "
        f"ASR {s['Final_ASR']:.2f}% | "
        f"BenignDrop {s['Avg_Benign_Drop']:.2f}% | "
        f"Queries {s['Avg_Queries']:.0f} | "
        f"Infer {s['Avg_Infer_s']:.3f}s | "
        f"EntMAD {s['Avg_EntMAD_s']:.5f}s"
    )

# --------------------- Plot MNIST Ablation ASR (color-only fix) ---------------------
plt.figure(figsize=(7, 4))

style_map = {
    "Attack only no defense": {"color": "black"},
    "Filtering only": {"color": "orange"},
    "Weighting only": {"color": "blue"},
    "Full BlackBoxGuard": {"color": "red"},
}

for s in summaries:
    h = s["History"]
    style = style_map[s["Setting"]]
    plt.plot(
        h["round"],
        np.array(h["asr"]) * 100,
        label=s["Setting"],
        linewidth=2,
        **style
    )

plt.xlabel("Round")
plt.ylabel("ASR (%)")
plt.title("MNIST Ablation: ASR")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mnist_ablation_asr.png", dpi=200)

print("Saved: mnist_ablation_asr.png")

