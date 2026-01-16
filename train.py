import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from nlp_moe_model import MoEBertModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'   # AMD ROCm
# Custom Dataset for News
class NewsDataset(Dataset):
    def __init__(self, texts, labels, sentiments, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentiments = sentiments
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        sentiment = self.sentiments[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': t.tensor(label, dtype=t.long),
            'sentiment': t.tensor(sentiment, dtype=t.long)
        }

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_encoder = LabelEncoder()
    sentiment_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['Class Index'])
    test_df['label_encoded'] = label_encoder.transform(test_df['Class Index'])
    train_df['Sentiment_encoded'] = sentiment_encoder.fit_transform(train_df['Sentiment'])
    test_df['Sentiment_encoded'] = sentiment_encoder.transform(test_df['Sentiment'])
    return train_df, test_df, label_encoder, sentiment_encoder

def train_epoch(model, dataloader, optimizer, criterion, device, load_balance=False,balance_coef=0.01):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    epoch_entropy = []
    epoch_expert_usage = []
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        sentiment = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        if load_balance:
            # Get routing weights
            gate_logits = model.router(input_ids, attention_mask)
            gate_probs = t.softmax(gate_logits, dim=-1)

            avg_expert_usage = gate_probs.mean(dim=0)
            uniform_target = t.ones_like(avg_expert_usage) / avg_expert_usage.size(0)
            balance_loss = t.nn.functional.mse_loss(avg_expert_usage, uniform_target)

            # Entropy
            entropy = -t.sum(gate_probs * t.log(gate_probs + 1e-10), dim=-1).mean()
            balance_loss += 0.1 * entropy
            aux_loss = balance_loss - 0.1 * entropy
            loss = loss + balance_coef * aux_loss

            epoch_entropy.append(entropy.item())
            epoch_expert_usage.append(avg_expert_usage.detach().cpu().numpy())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    metrics = {
        'avg_entropy': np.mean(epoch_entropy) if epoch_entropy else 0,
        'expert_usage': np.mean(epoch_expert_usage, axis=0) if epoch_expert_usage else None
    }
    return total_loss / len(dataloader), correct / total, metrics

def evaluate(model, dataloader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with t.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    if return_predictions:
        return total_loss / len(dataloader), correct / total, all_predictions, all_labels
    return total_loss / len(dataloader), correct / total
