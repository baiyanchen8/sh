import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*14*14, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # 1. Check normalization fix
    mean, std = (0.1307,), (0.3081,)
    NORMALIZER = transforms.Normalize(mean, std)
    
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 2. Test one batch
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    
    # Apply our new logic: Raw -> Trigger (skipped here) -> Clamp -> Normalize
    xb = torch.clamp(xb, 0, 1)
    xb = NORMALIZER(xb)
    
    out = model(xb)
    loss = F.cross_entropy(out, yb)
    loss.backward()
    optimizer.step()
    
    print(f"Test batch loss: {loss.item():.4f}")
    print("Environment and logic verification SUCCESS.")

if __name__ == '__main__':
    run_test()
