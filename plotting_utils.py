import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from nlp_moe_model import MoEBertModel
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from train import train_epoch, evaluate, NewsDataset,load_data
import os
from config import *



def plot_ablation_results_v2(results_df, save_path='metrics/ablation_study.png'):
    """Visualize ablation study results - works with both MoE and dense models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Accuracy by config
    ax1 = axes[0]
    colors = ['steelblue' if t == 'moe' else 'coral' for t in results_df['type']]
    bars = ax1.bar(range(len(results_df)), results_df['test_acc'], color=colors)
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels(results_df['config_name'], rotation=45, ha='right')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy by Configuration')
    ax1.set_ylim([0, 1])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='MoE'),
                       Patch(facecolor='coral', label='Dense')]
    ax1.legend(handles=legend_elements)

    # 2. Accuracy vs Parameters (efficiency plot)
    ax2 = axes[1]
    for model_type in results_df['type'].unique():
        subset = results_df[results_df['type'] == model_type]
        ax2.scatter(subset['trainable_params'], subset['test_acc'],
                   label=model_type, s=100, alpha=0.7)
        # Add labels
        for _, row in subset.iterrows():
            ax2.annotate(row['config_name'],
                        (row['trainable_params'], row['test_acc']),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.set_xlabel('Trainable Parameters')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Accuracy vs Model Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_ablation_results(results_df):
    """Visualize ablation study results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Accuracy by config
    ax1 = axes[0]
    results_df['config_name'] = results_df.apply(
        lambda x: f"E{x['num_experts']}_{x['routing']}_d{x['dropout']}", axis=1
    )
    bars = ax1.bar(range(len(results_df)), results_df['test_acc'])
    ax1.set_xticks(range(len(results_df)))
    ax1.set_xticklabels(results_df['config_name'], rotation=45, ha='right')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy by Configuration')

    # 2. Specialization by num_experts
    ax2 = axes[1]
    for routing in results_df['routing'].unique():
        subset = results_df[results_df['routing'] == routing]
        ax2.plot(subset['num_experts'], subset['specialization'], 'o-', label=routing)
    ax2.set_xlabel('Number of Experts')
    ax2.set_ylabel('Specialization Score')
    ax2.set_title('Expert Specialization vs Num Experts')
    ax2.legend()

    # 3. Entropy by routing type
    ax3 = axes[2]
    routing_entropy = results_df.groupby('routing')['avg_entropy'].mean()
    ax3.bar(routing_entropy.index, routing_entropy.values)
    ax3.set_ylabel('Average Gating Entropy')
    ax3.set_title('Gating Entropy by Routing Type')

    plt.tight_layout()
    plt.savefig('metrics/ablation_study.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_expert_entropy(expert_usage, save_path='metrics/expert_entropy.png', hyperparams=None):
    """Shows how specialized (low entropy) or general (high entropy) each expert is"""
    # Normalize each expert's usage across classes
    expert_probs = expert_usage / (expert_usage.sum(axis=0, keepdims=True) + 1e-10)

    # Calculate entropy for each expert
    num_experts = expert_usage.shape[1]
    num_classes = expert_usage.shape[0]
    max_entropy = np.log(num_classes)  # Maximum entropy (uniform distribution)

    entropies = []
    for expert_idx in range(num_experts):
        probs = expert_probs[:, expert_idx]
        # Handle case where expert is never used
        if probs.sum() < 1e-10:
            entropy = max_entropy  # Treat unused expert as maximally uncertain
        else:
            # Calculate entropy, avoiding log(0)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)

    entropies = np.array(entropies)
    normalized_entropies = entropies / max_entropy  # Normalize to [0, 1]

    # Debug print
    print(f"Expert entropies (raw): {entropies}")
    print(f"Expert entropies (normalized): {normalized_entropies}")
    print(f"Expert usage sum per expert: {expert_usage.sum(axis=0)}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if e < 0.5 else 'blue' for e in normalized_entropies]
    bars = ax.bar(range(num_experts), normalized_entropies, color=colors)

    ax.set_xlabel('Expert')
    ax.set_ylabel('Normalized Entropy')
    ax.set_title('Expert Specialization (Low = Specialist, High = Generalist)')
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f'Expert {i}' for i in range(num_experts)])
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Specialist/Generalist threshold')
    ax.legend()
    ax.set_ylim([0, 1.1])

    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, hyperparams, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_confusion_matrix(y_true, y_pred, class_names, save_path='metrics/confusion_matrix.png', hyperparams=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    fig = plt.gcf()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # Add hyperparameters text box
    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")
def compute_expert_preference(expert_usage):
    """Normalize expert usage to show relative preference per class"""
    # Normalize each expert column (across classes)
    col_normalized = expert_usage / (expert_usage.sum(axis=0, keepdims=True) + 1e-10)
    return col_normalized

def plot_expert_preference_heatmap(expert_usage, class_names, save_path='metrics/expert_preference.png', hyperparams=None):
    """Shows which classes prefer which experts (column-normalized)"""
    preference = compute_expert_preference(expert_usage)

    plt.figure(figsize=(12, max(8, len(class_names) * 0.4)))
    sns.heatmap(preference,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                xticklabels=[f'Expert {i}' for i in range(expert_usage.shape[1])],
                yticklabels=class_names,
                vmin=0, vmax=0.5,
                cbar_kws={'label': 'Class Preference (normalized)'})

    fig = plt.gcf()
    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)

    plt.title('Expert Preference by Class (Column Normalized)', fontsize=14, fontweight='bold')
    plt.xlabel('Experts')
    plt.ylabel('Classes')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_training_history(train_losses, train_accs, test_losses, test_accs, save_path='metrics/training_history.png', hyperparams=None):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Add hyperparameters text box
    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to {save_path}")

def plot_per_class_metrics(y_true, y_pred, class_names, save_path='metrics/per_class_metrics.png', hyperparams=None):
    """Plot per-class precision, recall, and F1-score"""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add hyperparameters text box
    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Per-class metrics saved to {save_path}")

def analyze_expert_usage(model, dataloader, device, num_experts, num_classes):
    """Analyze which experts are used for each class"""
    model.eval()

    # Track expert usage per class: [num_classes, num_experts]
    expert_usage = np.zeros((num_classes, num_experts))
    class_counts = np.zeros(num_classes)

    with t.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing expert usage"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Get BERT embeddings
            bert_output = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_output.pooler_output

            # Get routing weights from the MoE layer
            routing_weights = model.router(input_ids=input_ids, attention_mask=attention_mask)  # [batch_size, num_experts]

            # For soft routing, use the weights directly
            # For hard/top-k routing, you'll need to adapt based on your implementation
            if model.routing == 'soft':
                expert_probs = t.softmax(routing_weights, dim=-1)
            else:
                # For top-k, get which experts were selected
                top_k_values, top_k_indices = t.topk(routing_weights, model.top_k, dim=-1)
                expert_probs = t.zeros_like(routing_weights)
                expert_probs.scatter_(1, top_k_indices, t.softmax(top_k_values, dim=-1))

            # Accumulate expert usage per class
            for i, label in enumerate(labels):
                label_idx = label.item()
                expert_usage[label_idx] += expert_probs[i].cpu().numpy()
                class_counts[label_idx] += 1

    # Normalize by class counts
    for i in range(num_classes):
        if class_counts[i] > 0:
            expert_usage[i] /= class_counts[i]

    return expert_usage

def plot_expert_usage_heatmap(expert_usage, class_names, save_path='metrics/expert_usage_heatmap.png', hyperparams=None):
    """Plot heatmap showing expert usage per class"""
    plt.figure(figsize=(12, max(8, len(class_names) * 0.4)))

    sns.heatmap(expert_usage,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=[f'Expert {i}' for i in range(expert_usage.shape[1])],
                yticklabels=class_names,
                cbar_kws={'label': 'Average Usage Weight'})
    fig = plt.gcf()
    # Add hyperparameters text box
    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)


    plt.title('Expert Usage by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Experts')
    plt.ylabel('Classes')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Expert usage heatmap saved to {save_path}")

def plot_expert_specialization(expert_usage, class_names, save_path='metrics/expert_specialization.png',hyperparams=None):
    """Plot which classes each expert specializes in"""
    num_experts = expert_usage.shape[1]

    fig, axes = plt.subplots(2, (num_experts + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()

    for expert_idx in range(num_experts):
        ax = axes[expert_idx]
        usage = expert_usage[:, expert_idx]

        # Sort classes by usage
        sorted_indices = np.argsort(usage)[::-1][:5]  # Top 5 classes

        ax.barh(range(len(sorted_indices)), usage[sorted_indices])
        ax.set_yticks(range(len(sorted_indices)))
        ax.set_yticklabels([class_names[i] for i in sorted_indices], fontsize=8)
        ax.set_xlabel('Usage Weight', fontsize=9)
        ax.set_title(f'Expert {expert_idx}', fontsize=10, fontweight='bold')
        ax.invert_yaxis()

    # Hide extra subplots if odd number of experts
    for i in range(num_experts, len(axes)):
        axes[i].axis('off')

    # Add hyperparameters text box
    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Expert specialization saved to {save_path}")

def plot_gating_entropy_evolution(entropy_per_epoch, save_path='metrics/gating_entropy_evolution.png'):
    """Plot how gating entropy changes during training"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(entropy_per_epoch) + 1), entropy_per_epoch, 'b-o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Gating Entropy')
    plt.title('Evolution of Gating Entropy During Training')
    plt.grid(True, alpha=0.3)

    # Add interpretation
    if entropy_per_epoch[-1] < entropy_per_epoch[0]:
        plt.annotate('Experts specializing', xy=(len(entropy_per_epoch), entropy_per_epoch[-1]),
                    fontsize=10, color='green')
    else:
        plt.annotate('Experts staying general', xy=(len(entropy_per_epoch), entropy_per_epoch[-1]),
                    fontsize=10, color='orange')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_expert_usage_evolution(expert_usage_history, save_path='metrics/expert_usage_evolution.png', hyperparams=None):
    """Plot how expert usage changes during training epochs"""
    expert_usage_history = np.array(expert_usage_history)  # Shape: [num_epochs, num_experts]
    num_epochs, num_experts = expert_usage_history.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Line plot: Expert usage over epochs
    ax1 = axes[0]
    for expert_idx in range(num_experts):
        ax1.plot(range(1, num_epochs + 1), expert_usage_history[:, expert_idx],
                 'o-', label=f'Expert {expert_idx}', linewidth=2, markersize=4)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Usage Weight')
    ax1.set_title('Expert Usage Evolution During Training')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Add uniform line for reference
    uniform_usage = 1.0 / num_experts
    ax1.axhline(y=uniform_usage, color='gray', linestyle='--', alpha=0.7, label='Uniform')

    # 2. Heatmap: Expert usage across epochs
    ax2 = axes[1]
    sns.heatmap(expert_usage_history.T,
                annot=True if num_epochs <= 10 else False,
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=[f'E{i+1}' for i in range(num_epochs)],
                yticklabels=[f'Expert {i}' for i in range(num_experts)],
                ax=ax2,
                cbar_kws={'label': 'Usage Weight'})
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Expert')
    ax2.set_title('Expert Usage Heatmap Over Epochs')

    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Expert usage evolution saved to {save_path}")


def plot_expert_usage_std_evolution(expert_usage_history, save_path='metrics/expert_usage_std.png',hyperparams=None):
    """Plot the standard deviation of expert usage (measures balance) over epochs"""
    expert_usage_history = np.array(expert_usage_history)

    # Compute std across experts for each epoch
    usage_std = expert_usage_history.std(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(usage_std) + 1), usage_std, 'r-o', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Std Dev of Expert Usage')
    plt.title('Expert Load Balance Over Training\n(Lower = More Balanced)')
    plt.grid(True, alpha=0.3)

    # Add interpretation
    if usage_std[-1] < usage_std[0]:
        plt.annotate('Becoming more balanced', xy=(len(usage_std), usage_std[-1]),
                    fontsize=10, color='green')
    else:
        plt.annotate('Becoming more imbalanced', xy=(len(usage_std), usage_std[-1]),
                    fontsize=10, color='red')
    if hyperparams:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig = plt.gcf()
        fig.text(0.02, 0.98, hyperparams, transform=fig.transFigure,
                 fontsize=9, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def get_hyperparameters_text():
    return f"""Hyperparameters:
Batch Size: {BATCH_SIZE}
Epochs: {EPOCHS}
Learning Rate: {LEARNING_RATE}
Number of Experts: {NUM_EXPERTS}
Model Name: {MODEL_NAME}
Max Length: {MAX_LEN}
Routing: {ROUTING}
Top K: {TOP_K}
Freeze BERT: {FREEZE_BERT}
Load Balancing: {LOAD_BALANCE}
Load Balancing Coefficient: {LOAD_BALANCE_COEF}
"""