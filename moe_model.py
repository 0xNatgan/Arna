import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class BasicExpert(nn.Module):
    def __init__(self,feature_in,feature_out):
        super().__init__()
        self.layer = nn.Linear(feature_in,feature_out)
        self.activation = nn.ReLU()

    def forward(self,x):
        return self.activation(self.layer(x))

class MoEModel(nn.Module):
    def __init__(self,feature_in,feature_out,expert_number):
        super().__init__()
        self.experts = nn.ModuleList(
            [BasicExpert(feature_in,feature_out) for _ in range(expert_number)]
        )
        self.gating_network = nn.Linear(feature_in, expert_number)


class TextExpert(nn.Module):
    def __init__(self, hidden_dim, output_dim,dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class MoETextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, expert_number,num_heads=4,num_layers=2,freeze_bert=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = embedding_dim,
            nhead = num_heads,
            dim_feedforward = hidden_dim,
            batch_first = True
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.experts = nn.ModuleList(
            [TextExpert(hidden_dim, output_dim) for _ in range(expert_number)]
        )
        self.gating_network = nn.Linear(embedding_dim, expert_number)

    def forward(self, x):
        seq_len = x.size(1)
        embedded = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        encoded = self.encoder(embedded)
        pooled = encoded.mean(dim=1)

        gating_weights = t.softmax(self.gating_network(pooled), dim=1)
        expert_outputs = t.stack([expert(pooled) for expert in self.experts], dim=1)
        output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return output

class MoEBertModel(nn.Module): # MoE model using BERT as the base model for NEWS
    def __init__(self, pretrained_model_name, expert_number, output_dim):
        super().__init__()
        self.bert = tf.BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        self.experts = nn.ModuleList(
            [TextExpert(hidden_size, output_dim) for _ in range(expert_number)]
        )
        self.gating_network = nn.Linear(hidden_size, expert_number)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output

        gating_weights = t.softmax(self.gating_network(pooled_output), dim=1)
        expert_outputs = t.stack([expert(pooled_output) for expert in self.experts], dim=1)
        output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return output