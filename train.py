import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from nlp_moe_model import MoEBertModel, MoETextModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Seul GPU 0 visible
os.environ['HIP_VISIBLE_DEVICES'] = '0'   # Pour AMD ROCm
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

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        sentiment = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

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

    return total_loss / len(dataloader), correct / total

def main(model_type='bert'):
    # Configuration
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPU Name: {t.cuda.get_device_name(0)}")
    print(f"GPU Count: {t.cuda.device_count()}")

    # Forcer toutes les opérations sur ce GPU
    t.cuda.set_device(0)

    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    NUM_EXPERTS = 10
    MODEL_NAME = 'bert-base-uncased'
    MAX_LEN = 128
    # Load data
    train_df, test_df, label_encoder, sentiment_encoder = load_data(
        '/home/knwldgosint/Documents/School5/Advanced Neural network/project/Arna/dataset/train_with_sentiment.csv',
        '/home/knwldgosint/Documents/School5/Advanced Neural network/project/Arna/dataset/test_with_sentiment.csv'
    )
    num_classes = len(label_encoder.classes_)
    num_sentiments = len(sentiment_encoder.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Number of sentiment classes: {num_sentiments}")


    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets
    train_dataset = NewsDataset(
        train_df['Description'].tolist(),
        train_df['label_encoded'].tolist(),
        train_df['Sentiment_encoded'].tolist(),
        tokenizer,
        max_length=MAX_LEN
    )
    test_dataset = NewsDataset(
        test_df['Description'].tolist(),
        test_df['label_encoded'].tolist(),
        test_df['Sentiment_encoded'].tolist(),
        tokenizer,
        max_length=MAX_LEN
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    model = None
    # Model
    # if model_type == 'bert':
    model = MoEBertModel(
        pretrained_model_name=MODEL_NAME,
        expert_number=NUM_EXPERTS,
        output_dim=num_classes,
        routing = 'gumbel',
        top_k = 3,
        freeze_bert = True
    ).to(device)
    # elif model_type == 'text':
    #     model = MoETextModel(
    #         vocab_size=tokenizer.vocab_size,
    #         embedding_dim=128,
    #         hidden_dim=256,
    #         output_dim=num_classes,
    #         expert_number=NUM_EXPERTS
    #     ).to(device)
    # else:
    #     raise ValueError("Invalid model type. Choose 'bert' or 'text'.")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_acc = 0
    test_acc=0
    test_loss=0
    train_loss=0
    train_acc=0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            t.save(model.state_dict(), 'best_moe_model.pth')
            print("Model saved!")
    metrics = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }
    t.save(metrics, 'training_metrics.pth')
    print(f"\nBest Test Accuracy: {best_acc:.4f}")

    t.cuda.empty_cache()

if __name__ == "__main__":
    main()
