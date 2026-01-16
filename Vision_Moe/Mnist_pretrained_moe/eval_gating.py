import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from .transforms import get_transforms
from .mnist_cnn import MNIST_CNN
from .gating_Mnist import MixtureOfExperts

def load_dense_model(model_path, device):
    """Load the dense expert model."""
    model = MNIST_CNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def load_moe_model(expert_paths, moe_path, device):
    """Load the MoE model with trained gating network."""
    moe = MixtureOfExperts(expert_paths)
    moe.load_state_dict(torch.load(moe_path))
    moe.to(device)
    moe.eval()
    return moe

def evaluate_model(model, image, is_moe=False):
    """Evaluate a single image and return the predicted class and expert weights (if MoE)."""
    model.eval()
    with torch.no_grad():
        if is_moe:
            output, gate_weights = model(image.unsqueeze(0))
            predicted_class = torch.argmax(output, dim=1).item()
            return predicted_class, gate_weights.squeeze(0).cpu().numpy()
        else:
            output = model(image.unsqueeze(0))
            predicted_class = torch.argmax(output, dim=1).item()
            return predicted_class , None


def evaluate_gating(batch_size=64, epochs=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    expert_paths = {
        'normal': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_normal.pth',
        'angle': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_angle.pth',
        'noise': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_noise.pth',
        'blur': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_blur.pth'
    }

    dense_model_path = './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_dense.pth'
    moe_model_path = './Mnist_pretrained_moe/Mnist_pretrained_experts/moe_gating_model.pth'

    print("Loading models...")
    moe_model = load_moe_model(expert_paths, moe_model_path, device)
    dense_model = load_dense_model(dense_model_path, device)
    print("Models loaded successfully!")

    transform_list = [get_transforms(expert_type) for expert_type in expert_paths.keys()]
    transform = transforms.RandomChoice(transform_list)

    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("test_loader loaded successfully!", len(test_loader.dataset), "test samples")

    moe_model.eval()
    dense_model.eval()

    for epoch in range(epochs):
        moe_correct = 0
        dense_correct = 0
        total = 0
        expert_usage = np.zeros(len(expert_paths))
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total += batch_size

            # Evaluate MoE model
            with torch.no_grad():
                outputs, gate_weights = moe_model(data)
                _, moe_predicted = torch.max(outputs.data, 1)
                moe_correct += (moe_predicted == target).sum().item()
                expert_usage += gate_weights.sum(dim=0).cpu().numpy()

            # Evaluate Dense model
            with torch.no_grad():
                outputs = dense_model(data)
                _, dense_predicted = torch.max(outputs.data, 1)
                dense_correct += (dense_predicted == target).sum().item()

        moe_accuracy = 100 * moe_correct / total
        dense_accuracy = 100 * dense_correct / total
        expert_usage /= total

    print(f"")
    print(f"MoE Accuracy: {moe_accuracy:.2f}%, Dense Accuracy: {dense_accuracy:.2f}%")
    usage_str = ', '.join([f' {list(expert_paths.keys())[i]}: {100.0 * usage:.1f}%' for i, usage in enumerate(expert_usage)])
    print(f'Expert Usage Distribution: {usage_str}')
    return moe_accuracy, dense_accuracy, expert_usage

def evaluate_label_filter(label_filter=None, alteration=None, batch_size=256, epochs=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_paths = {
        'normal': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_normal.pth',
        'angle': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_angle.pth',
        'noise': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_noise.pth',
        'blur': './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_blur.pth'
    }

    dense_model_path = './Mnist_pretrained_moe/Mnist_pretrained_experts/expert_dense.pth'
    moe_model_path = './Mnist_pretrained_moe/Mnist_pretrained_experts/moe_gating_model.pth'

   
    moe_model = load_moe_model(expert_paths, moe_model_path, device)
    dense_model = load_dense_model(dense_model_path, device)

    transform = get_transforms(alteration) 


    test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
    indices = torch.where(test_dataset.targets == label_filter)[0]
    filtered_dataset = Subset(test_dataset, indices)

    
    test_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

    moe_model.eval()
    dense_model.eval()

    for epoch in range(epochs):
        moe_correct = 0
        dense_correct = 0
        total = 0
        expert_usage = np.zeros(len(expert_paths))
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total += batch_size

            # Evaluate MoE model
            with torch.no_grad():
                outputs, gate_weights = moe_model(data)
                _, moe_predicted = torch.max(outputs.data, 1)
                moe_correct += (moe_predicted == target).sum().item()
                expert_usage += gate_weights.sum(dim=0).cpu().numpy()

        moe_accuracy = 100 * moe_correct / total
        expert_usage /= total

    return expert_usage
