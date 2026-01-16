import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from .transforms import get_transforms
from .mnist_cnn import MNIST_CNN

class VisionMoE(nn.Module):
    def __init__(self, experts, num_classes=10):
        super(VisionMoE, self).__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        
        # Gating network: Takes image, outputs weights for each expert
        # Input: Flattened image (28*28) or processed feature
        self.gating_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 1. Get gating weights
        weights = self.gating_network(x) # [batch_size, num_experts]
        
        # 2. Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) # [batch, num_experts, num_classes]
        
        # 3. Combine
        # weights.unsqueeze(2) becomes [batch, num_experts, 1]
        # We multiply expert outputs by their weights and sum across experts
        weighted_output = torch.sum(expert_outputs * weights.unsqueeze(2), dim=1)
        
        return weighted_output, weights

def train_experts(epochs=5, save_dir=None, verbose=False):
    experts_to_train = ['normal', 'angle', 'noise', 'blur']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    accuracies = {}
    for expert_name in experts_to_train:
        accuracies.update(train_one_expert(expert_name, device, epochs=epochs, save_path=save_dir, verbose=verbose))
    return accuracies

def train_dense_expert(epochs=5, save_dir=None, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return train_one_expert('dense', device, epochs=epochs, save_path=save_dir, verbose=verbose)    

def train_one_expert(expert_name, device, epochs=5, save_path=None, verbose=False):
    print(f"\n{'='*20}")
    print(f"Training Expert: {expert_name}")
    print(f"{'='*20}")
    
    # Get specific transforms
    transform = get_transforms(expert_name)
    
    # Load Data (download to ./dataset folder)
    data_path = './dataset'
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    # We use test loader with the SAME transform to test expert performance on its specific domain
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    model = MNIST_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (batch_idx+1) % 100 == 0:
                if verbose:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = 100 * correct / total
    print(f"Accuracy for {expert_name}: {acc:.2f}%")
    
    # Save
    if save_path is None:
        
        save_path = f"./Mnist_pretrained_moe/Mnist_pretrained_experts/expert_{expert_name}.pth"
    else :
        save_path = save_path + f"expert_{expert_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved {expert_name} model to {save_path}")

    return { expert_name:acc }
