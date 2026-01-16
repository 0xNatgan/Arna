from pyexpat import model
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


class TextExpert2(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # First block with residual
        h = self.act(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        h = h + x  # Residual connection
        h = self.act(h)
        return self.fc_out(h)

class TextExpert(nn.Module):
    def __init__(self, hidden_dim, output_dim,dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

# class MoETextModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, expert_number,num_heads=4,num_layers=2,freeze_bert=False):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model = embedding_dim,
#             nhead = num_heads,
#             dim_feedforward = hidden_dim,
#             batch_first = True
#         )
#         if freeze_bert:
#             for param in self.bert.parameters():
#                 param.requires_grad = False
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.experts = nn.ModuleList(
#             [TextExpert(hidden_dim, output_dim) for _ in range(expert_number)]
#         )
#         self.gating_network = nn.Linear(embedding_dim, expert_number)

#     def forward(self, x):
#         seq_len = x.size(1)
#         embedded = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
#         encoded = self.encoder(embedded)
#         pooled = encoded.mean(dim=1)

#         gating_weights = t.softmax(self.gating_network(pooled), dim=1)
#         expert_outputs = t.stack([expert(pooled) for expert in self.experts], dim=1)
#         output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)
#         return output

class MoEBertModel(nn.Module): # MoE model using BERT as the base model for NEWS
    # Maybe will add noise_std later
    def __init__(self, pretrained_model_name, expert_number, output_dim,top_k=None,
                 routing='soft',tau=1.0, freeze_bert=False,gumbel_hard=False,dropout=0.3):
        super().__init__()
        self.bert = tf.BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.routing = routing
        self.gumbel_hard = gumbel_hard
        self.tau = tau
        self.expert_number = expert_number
        self.top_k = top_k if top_k is not None else expert_number  # Use all experts if top_k not specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        # self.experts = nn.ModuleList(
        #     [TextExpert(hidden_size, output_dim) for _ in range(expert_number // 2)] +
        #     [TextExpert2(hidden_size, output_dim) for _ in range(expert_number - expert_number // 2)]
        # )
        self.experts=nn.ModuleList([TextExpert2(hidden_size, output_dim,dropout) for _ in range(expert_number)])
        self.gating_network = nn.Linear(hidden_size, expert_number)
        # Testing initialization of gating network in order to balance expert usage
        nn.init.xavier_uniform_(self.gating_network.weight)
        nn.init.constant_(self.gating_network.bias, 0.0)


    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        gate = self.gating_network(pooled_output)

        if self.routing == 'soft':
            # Standard soft routing
            gating_weights = t.softmax(gate, dim=1)
            expert_outputs = t.stack([expert(pooled_output) for expert in self.experts], dim=1)
            output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'hard':
            # Hard routing
            gating_weights = t.softmax(gate, dim=1)
            top_k_weights, top_k_indices = t.topk(gating_weights, self.top_k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)

            # Compute all expert outputs at once
            expert_outputs = t.stack([expert(pooled_output) for expert in self.experts], dim=1)

            # Create sparse weight matrix
            batch_size = pooled_output.size(0)
            sparse_weights = t.zeros(batch_size, len(self.experts), device=gate.device)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)

            # Weighted sum
            output = t.bmm(sparse_weights.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'gumbel':
            # Gumbel-Softmax
            gating_weights = nn.functional.gumbel_softmax(gate, tau=self.tau, hard=self.gumbel_hard, dim=1)
            expert_outputs = t.stack([expert(pooled_output) for expert in self.experts], dim=1)
            output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)


        return output
    def router(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        routing_weights = self.gating_network(pooled_output)
        return routing_weights

class BertDenseBaseline(nn.Module):
    """Baseline: BERT + single dense head (no MoE)"""
    def __init__(self, pretrained_model_name, output_dim, hidden_multiplier=1, freeze_bert=False, dropout=0.3):
        super().__init__()
        self.bert = tf.BertModel.from_pretrained(pretrained_model_name)
        hidden_dim = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Single dense head with equivalent capacity to MoE experts
        # If MoE has 8 experts with hidden_dim each, this has hidden_dim * hidden_multiplier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * hidden_multiplier),
            nn.LayerNorm(hidden_dim * hidden_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * hidden_multiplier, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.classifier(pooled)