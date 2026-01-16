import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from numpy.random import choice
import numpy as np
from .transforms import get_transforms
from .MoE_model import MoE_Model
from .base_models import MNIST_CNN1, MNIST_CNN2, MNIST_CNN3
from .gating_model import GatingNetwork




def create_moe_model(nb_experts=4, k_training=2, k_inference=1, tau=0.8, lambda_balance=0.05):
    """
    Factory function to create a MoE model using the CNN experts defined in base_models.
    """
    expert_list = [MNIST_CNN1, MNIST_CNN2, MNIST_CNN3]
    experts = []
    for _ in range(nb_experts):
        experts.append(choice(expert_list)())
    gating_network = GatingNetwork(num_experts=nb_experts)
    
    return MoE_Model(
        experts=experts,
        gating_network=gating_network,
        k_training=k_training,
        k_inference=k_inference,
        tau=tau,
        lambda_balance=lambda_balance
    )

def train_moe_one_epoch(model: MoE_Model, 
                        train_loader, 
                        optimizer, 
                        criterion, 
                        device, 
                        k=None, 
                        selection_policy='hard', 
                        gumbell_softmax=True, 
                        threshold=None):
    """
    Standard training epoch handling the total loss (CrossEntropy + Load Balance).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass returns prediction and auxiliary balancing loss
        output, aux_loss = model.forward(data,k=k, selection_policy=selection_policy, gumbell_softmax=gumbell_softmax, threshold=threshold)
        
        # Calculate primary classification loss
        main_loss = criterion(output, target)
        
        # Total loss = Classification loss + Balancing loss
        total_loss = main_loss + aux_loss
        
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        
        # Calculate accuracy for monitoring
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate_moe(model: MoE_Model, test_loader, criterion, device, k=None, selection_policy='hybrid', gumbell_softmax=True, verbose=False):
    """
    Evaluation function using sparse inference.
    Optionally returns the repartition of experts called during the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Initialize repartition counter on device for speed
    repartition = np.zeros(len(model.experts), dtype=int)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)    
            res = model.inference(data, k=k, selection_policy=selection_policy ,gumbell_softmax=gumbell_softmax, verbose=verbose)
            
            if verbose:
                output, probs = res
                repartition += (probs > 0).sum(dim=0).cpu().numpy()
            else:
                output = res
            
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    if verbose:
        return avg_loss, accuracy, repartition / repartition.sum()
    else:
        return avg_loss, accuracy

def moe_train_loop(model, epochs=10, k=None, selection_policy_t='soft', selection_policy_i='hybrid', gumbell_softmax_t=True, gumbell_softmax_i=True, threshold=None):
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Data loading
    transform = get_transforms()
    data_path = './dataset'
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


    print(f"\nStarting MoE training...")
    print(f"Number of Experts: {len(model.experts)} | k_training: {model.k_training} | k_inference: {model.k_inference}\n")
    print(f"experts: {model.experts_infos()}\n")
    epoch_informations = {}
    
    
    for epoch in range(epochs):
        train_loss, train_acc = train_moe_one_epoch(model, train_loader, optimizer, criterion, device, k=model.k_training, selection_policy=selection_policy_t, gumbell_softmax=gumbell_softmax_t, threshold=threshold)
        test_loss, test_acc, repartition = evaluate_moe(model, test_loader, criterion, device, k=model.k_inference, selection_policy=selection_policy_i, gumbell_softmax=gumbell_softmax_i, verbose=True)
        epoch_informations[epoch] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "repartition": repartition,
            "lambda_balance": model.lambda_balance
        }
        formatted_repartition = [f"{x:.3f}" for x in repartition.tolist()]
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
              f"\nExpert Repartition: {formatted_repartition}\n",
              f"Lambda Balance: {model.lambda_balance:.6f}\n")
        

    save_path = "./Mnist_MoE/Mnist_MoE_experts/expert_moe.pth"
    torch.save(model.state_dict(), save_path)
    print(f"MoE model saved to {save_path} with accuracy {test_acc:.2f}%\n")
    return {"MoE_Model": test_acc}, epoch_informations

def moe_eval_on_class(model, k=None, selection_policy='hard', gumbell_softmax=True, threshold=None):
    """
    Evaluate the MoE model on each class separately to see expert specialization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = get_transforms()
    data_path = './dataset'
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    criterion = torch.nn.CrossEntropyLoss()

    class_informations = {}

    for class_label in range(10):
        # Filter dataset for the current class
        class_indices = [i for i, (_, label) in enumerate(test_dataset) if label == class_label]
        class_subset = torch.utils.data.Subset(test_dataset, class_indices)
        class_loader = DataLoader(dataset=class_subset, batch_size=64, shuffle=False)

        test_loss, test_acc, repartition = evaluate_moe(model, class_loader, criterion, device, k=model.k_inference, selection_policy=selection_policy, gumbell_softmax=gumbell_softmax, verbose=True)
        class_informations[class_label] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "repartition": repartition
        }
        print(f"Class {class_label} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"Expert Repartition: {repartition.tolist()}\n")

    return class_informations