import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .base_models import MNIST_MediumCNN
from .transforms import get_transforms

def train_dense_model(epochs=10, batch_size=64, lr=0.001):
    """
    Train a dense expert model on MNIST with various corruptions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training Dense Model for {epochs} epochs...")
    
    # Get dense transform
    transform = get_transforms()
    
    # Load datasets
    train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = MNIST_MediumCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting training...")
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/100:.4f}, Train Acc: {100*correct/total:.2f}%")
                running_loss = 0.0
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = 100 * test_correct / test_total
        print(f"\nEpoch {epoch+1} - Test Accuracy: {test_acc:.2f}%\n")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = "./Mnist_MoE/Mnist_MoE_experts/dense_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} with accuracy {test_acc:.2f}%\n")
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.2f}%")
    return {"Dense_Model": best_acc}

def eval_dense_model(model, test_loader):
    """
    Evaluate the dense expert model on the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
    
    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy, model