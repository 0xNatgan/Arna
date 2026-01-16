import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .mnist_cnn import MNIST_CNN
from .transforms import get_transforms

class GatingNetwork(nn.Module):
    """
    Gating network that learns to select the best expert for each input.
    """
    def __init__(self, num_experts=4):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)

class MixtureOfExperts(nn.Module):
    """
    MoE model that combines multiple expert models using a gating network.
    """
    def __init__(self, expert_paths, num_classes=10):
        super(MixtureOfExperts, self).__init__()
        
        # Load pretrained experts
        self.experts = nn.ModuleList()
        self.expert_names = []
        
        for name, path in expert_paths.items():
            expert = MNIST_CNN()
            expert.load_state_dict(torch.load(path))
            # Freeze expert parameters - we only train the gating network
            for param in expert.parameters():
                param.requires_grad = False
            expert.eval()
            self.experts.append(expert)
            self.expert_names.append(name)
        
        self.num_experts = len(self.experts)
        self.gating = GatingNetwork(num_experts=self.num_experts)
        self.num_classes = num_classes
    
    def forward(self, x):
        # Get gating weights [batch_size, num_experts]
        gate_weights = self.gating(x)
        
        # Get expert outputs [batch_size, num_experts, num_classes]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Weighted combination [batch_size, num_classes]
        output = torch.sum(expert_outputs * gate_weights.unsqueeze(2), dim=1)
        
        return output, gate_weights
    
    def output_top_k(self, data, k = 2):
        """Return the top k experts based on gating weights on the given data."""
        with torch.no_grad():
            _, gate_weights = self.gating(data)
            top_k_weights, top_k_indices = torch.topk(gate_weights, k, dim=1)
        return top_k_weights, top_k_indices

        

def train_gating_network(expert_paths, epochs=10, batch_size=64, lr=0.001, verbose=True):
    """
    Train the gating network to select the right expert for each input.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training MoE with experts: {list(expert_paths.keys())}")
    
    # Create mixed dataset with various corruptions
    transform_list = [get_transforms(expert_type) for expert_type in expert_paths.keys()]
    transform = transforms.RandomChoice(transform_list)
    
    train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create MoE model
    moe = MixtureOfExperts(expert_paths).to(device)
    
    # Only the gating network parameters are trainable
    optimizer = optim.Adam(moe.gating.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        moe.train()
        for expert in moe.experts:
            expert.eval()  # Keep experts in eval mode
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, gate_weights = moe(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                if verbose:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                          f"Loss: {running_loss/100:.4f}, Acc: {100*correct/total:.2f}%")
                running_loss = 0.0
        
        # Evaluation
        moe.eval()
        test_correct = 0
        test_total = 0
        expert_usage = torch.zeros(moe.num_experts)
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, gate_weights = moe(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
                
                # Track expert usage
                expert_selection = torch.argmax(gate_weights, dim=1)
                for i in range(moe.num_experts):
                    expert_usage[i] += (expert_selection == i).sum().item()
        
        test_acc = 100 * test_correct / test_total
        if verbose:
            print(f"\nEpoch {epoch+1} - Test Accuracy: {test_acc:.2f}%")
            print(f"Expert usage (%): ", end="")
            for i, name in enumerate(moe.expert_names):
                print(f"{name}: {expert_usage[i]/test_total*100:.1f}%  ", end="")
        print("\n" + "-" * 50)
    
    # Save the trained MoE model
    save_path = "./Mnist_pretrained_moe/Mnist_pretrained_experts/moe_gating_model.pth"
    torch.save(moe.state_dict(), save_path)
    print(f"\nTraining complete! Model saved to {save_path}")
    
    return moe, 