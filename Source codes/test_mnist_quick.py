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
import os, json


# ------------------- Transforms ---------------------
transform_train = transforms.Compose([transforms.ToTensor()])
transform_test  = transforms.Compose([transforms.ToTensor()])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- FL hyperparams ---------------------
NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 10
GLOBAL_ROUNDS = 1
LOCAL_EPOCHS = 2
BATCH_SIZE = 64

# Model & optimization
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# --------------------- BlackBoxGuard ---------------------
REF_SIZE = 2000           # server-side reference dataset size (not shared)
ALPHA = 2.0               # MAD multiplier
EPS = 1e-6

# Ablation switches 
USE_FILTERING = True      # set False for "weighting only"
USE_WEIGHTING = True      # set False for "filtering only"

# --------------------- Poisoning (attack settings) ----------------------
NUM_MALICIOUS = 5
POISON_RATE = 0.2
TRIGGER_VALUE = 1.0
TRIGGER_SIZE = 3
TARGET_LABEL = 0

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --------------------- Model ---------------------
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

# --------------------- Data ----------------------
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
testset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# Create balanced server-side reference dataset V (REF_SIZE) from trainset (disjoint, not shared)
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

# Split remaining indices into NUM_CLIENTS shards
shard_size = len(client_indices) // NUM_CLIENTS
clients_data = []
for ci in range(NUM_CLIENTS):
    start = ci * shard_size
    end = start + shard_size if ci < NUM_CLIENTS - 1 else len(client_indices)
    clients_data.append(client_indices[start:end])

ref_dataset = Subset(trainset, ref_indices)
ref_loader = DataLoader(ref_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# -------------------- Attack helpers ---------------------
def add_trigger(img_tensor, size=TRIGGER_SIZE, value=TRIGGER_VALUE):
    # img_tensor: (1,28,28)
    x = img_tensor.clone()
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

# --------------------- Training ---------------------
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

# --------------------- Defense helpers ---------------------
@torch.no_grad()
def get_prediction_probs(model_state_dict):
    """
    Server-side inference on the server-held reference dataset V.
    Returns probs with shape (REF_SIZE, num_classes).
    """
    m = SimpleCNN().to(device)
    m.load_state_dict(model_state_dict)
    m.eval()

    probs = []
    for xb, _ in ref_loader:
        xb = xb.to(device)
        logits = m(xb)
        p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
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
    # equal weights average
    return weighted_aggregate(state_dicts, [1.0] * len(state_dicts))

# --------------------- Evaluation (ACC / ASR) ---------------------
# Triggered test subset for ASR
triggered_images = []
for xb, _ in test_loader:
    for i in range(xb.size(0)):
        img = xb[i].clone()          # (1,28,28)
        img = add_trigger(img)       # (1,28,28)
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

# --------------------- Main FL loop (WITH DEFENSE) -----------------------
global_model = SimpleCNN().to(device)
global_state = copy.deepcopy(global_model.state_dict())

malicious_ids = set(random.sample(range(NUM_CLIENTS), NUM_MALICIOUS))
print("Malicious client ids:", malicious_ids)
print(f"Attack config: poison_rate={POISON_RATE}, attackers={NUM_MALICIOUS}, target={TARGET_LABEL}, trigger={TRIGGER_SIZE}x{TRIGGER_SIZE}")
print(f"Defense config: REF_SIZE={REF_SIZE}, ALPHA={ALPHA}, filtering={USE_FILTERING}, weighting={USE_WEIGHTING}")

history = {
    'round': [],
    'clean_acc': [],
    'asr': [],
    'benign_drop_rate': [],
    'server_queries': [],            # communication-style count
    't_infer_sec': [],               # timing: inference on V
    't_entropy_mad_sec': []          # timing: entropy + MAD computations
}

for rnd in trange(1, GLOBAL_ROUNDS + 1):
    selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)

    client_states = []
    entropies = []

    # Overhead counters (Comment 4)
    # Each selected client model is evaluated on REF_SIZE samples (server-side)
    server_queries = CLIENTS_PER_ROUND * REF_SIZE

    # 1) Local training + (server-side) prediction collection
    t0 = time.time()
    t_infer = 0.0

    for cid in selected:
        attacker = cid in malicious_ids
        data_tensor, label_tensor = make_client_dataset(clients_data[cid], attacker=attacker)

        local_model = SimpleCNN().to(device)
        local_model.load_state_dict(global_state)

        updated_state = local_train(local_model, data_tensor, label_tensor,
                                    epochs=LOCAL_EPOCHS, lr=LR, batch_size=BATCH_SIZE)
        client_states.append(updated_state)

        # Server-side inference on reference dataset (prediction-only observable)
        t_in0 = time.time()
        probs = get_prediction_probs(updated_state)
        t_infer += (time.time() - t_in0)

        entropies.append(mean_shannon_entropy(probs))

    # 2) Filtering (Median + MAD) + false positives 
    t1 = time.time()
    ent_arr = np.array(entropies, dtype=np.float64)
    med = float(np.median(ent_arr))
    mad = float(np.median(np.abs(ent_arr - med))) + 1e-12

    if USE_FILTERING:
        flags = np.abs(ent_arr - med) > ALPHA * mad
        retained_indices = [i for i, f in enumerate(flags) if not f]
        if len(retained_indices) == 0:
            retained_indices = list(range(len(client_states)))
            flags = np.array([False] * len(client_states), dtype=bool)
    else:
        flags = np.array([False] * len(client_states), dtype=bool)
        retained_indices = list(range(len(client_states)))

    num_filtered = int(flags.sum())

    benign_filtered = 0
    malicious_filtered = 0
    benign_selected = 0
    for i, cid in enumerate(selected):
        if cid in malicious_ids:
            if flags[i]:
                malicious_filtered += 1
        else:
            benign_selected += 1
            if flags[i]:
                benign_filtered += 1

    benign_drop_rate = (benign_filtered / benign_selected) if benign_selected > 0 else 0.0

    # 3) Weighting + aggregation
    retained_states = [client_states[i] for i in retained_indices]

    if USE_WEIGHTING:
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

    t_entropy_mad = time.time() - t1
    t_total = time.time() - t0  # not stored, but useful if you want

    # 4) Metrics each round
    clean_acc = evaluate_clean_acc(global_state)
    asr = compute_asr(global_state)

    history['round'].append(rnd)
    history['clean_acc'].append(clean_acc * 100)
    history['asr'].append(asr * 100)
    history['benign_drop_rate'].append(benign_drop_rate)
    history['server_queries'].append(server_queries)
    history['t_infer_sec'].append(t_infer)
    history['t_entropy_mad_sec'].append(t_entropy_mad)

    if rnd % 10 == 0:
        print(
            f"Round {rnd}: CA={clean_acc*100:.2f}% | ASR={asr*100:.2f}% | "
            f"filtered={num_filtered} (benign={benign_filtered}, mal={malicious_filtered}) | "
            f"benign_drop={benign_drop_rate*100:.2f}% | "
            f"queries={server_queries} | infer={t_infer:.2f}s | ent+mad={t_entropy_mad:.4f}s"
        )

# --------------------- Plots ---------------------
rounds = np.array(history['round'])
acc = np.array(history['clean_acc'])
asr = np.array(history['asr'])

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(rounds, acc)
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Clean Accuracy')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(rounds, asr, color='red')
plt.xlabel('Round')
plt.ylabel('Attack Success Rate (%)')
plt.title('Backdoor ASR')
plt.grid(True)

plt.tight_layout()
plt.savefig("mnist_defense_acc_asr.png", dpi=200)
print("Saved plot to mnist_defense_acc_asr.png")



# --------------------- Summary for overhead + false positives ---------------------
avg_queries = float(np.mean(history['server_queries']))
avg_infer = float(np.mean(history['t_infer_sec']))
avg_entmad = float(np.mean(history['t_entropy_mad_sec']))
avg_benign_drop = float(np.mean(history['benign_drop_rate'])) * 100

print(f"\n[Overhead Summary] avg server queries/round = {avg_queries:.0f}")
print(f"[Overhead Summary] avg inference time/round = {avg_infer:.3f} s")
print(f"[Overhead Summary] avg entropy+MAD time/round = {avg_entmad:.5f} s")
print(f"[False Positive Summary] avg benign drop rate = {avg_benign_drop:.2f}%")

os.makedirs("outputs", exist_ok=True)

out = {
    "dataset": "mnist",
    "attack": "badnets",
    "history": history
}

with open("outputs/mnist_badnets_blackboxguard.json", "w") as f:
    json.dump(out, f, indent=2)

print("Saved JSON to outputs/mnist_badnets_blackboxguard.json")


if __name__ == '__main__':

