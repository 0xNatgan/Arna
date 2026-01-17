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

# =======================================================
# Expert Networks
# =======================================================
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



# =======================================================
# MoE Components for Transformer V2
# =======================================================
class MoEFeedForward(nn.Module):
    """MoE layer that replaces the FFN in a Transformer layer"""
    def __init__(self, d_model, d_ff, num_experts, top_k=2, routing='hard', dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.routing = routing

        # Each expert is a simple FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        gate_logits = self.gate(x_flat)

        if self.routing == 'soft':
            # Soft routing - use all experts
            gate_probs = t.softmax(gate_logits, dim=-1)
            expert_outputs = t.stack([expert(x_flat) for expert in self.experts], dim=1)
            output = t.bmm(gate_probs.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'hard':
            # Hard routing - top-k experts only
            gate_probs = t.softmax(gate_logits, dim=-1)
            top_k_probs, top_k_indices = t.topk(gate_probs, self.top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            expert_outputs = t.stack([expert(x_flat) for expert in self.experts], dim=1)
            sparse_weights = t.zeros(x_flat.size(0), self.num_experts, device=x.device)
            sparse_weights.scatter_(1, top_k_indices, top_k_probs)
            output = t.bmm(sparse_weights.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'gumbel':
            # Gumbel-softmax routing
            gate_probs = nn.functional.gumbel_softmax(gate_logits, tau=1.0, hard=False, dim=-1)
            expert_outputs = t.stack([expert(x_flat) for expert in self.experts], dim=1)
            output = t.bmm(gate_probs.unsqueeze(1), expert_outputs).squeeze(1)

        else:
            raise ValueError(f"Unknown routing: {self.routing}")

        output = output.view(batch_size, seq_len, d_model)
        return output, t.softmax(gate_logits, dim=-1).view(batch_size, seq_len, -1)


class MoETransformerLayer(nn.Module):
    """Transformer layer with MoE replacing the FFN"""
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k=2, routing='hard', dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.moe_ffn = MoEFeedForward(d_model, d_ff, num_experts, top_k, routing, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        moe_output, gate_probs = self.moe_ffn(x)
        x = x + self.dropout2(moe_output)
        x = self.norm2(x)

        return x, gate_probs


# =======================================================
# Models
# =======================================================


class MoEBertModel(nn.Module):
    """BERT backbone with MoE classification head"""
    def __init__(self, pretrained_model_name, expert_number, output_dim, top_k=None,
                 routing='soft', tau=1.0, freeze_bert=False, gumbel_hard=False, dropout=0.3):
        super().__init__()
        self.bert = tf.BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.expert_number = expert_number
        self.routing = routing
        self.gumbel_hard = gumbel_hard
        self.tau = tau
        self.top_k = top_k if top_k is not None else expert_number

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.experts = nn.ModuleList([
            TextExpert2(self.hidden_size, output_dim, dropout) for _ in range(expert_number)
        ])
        self.gating_network = nn.Linear(self.hidden_size, expert_number)
        nn.init.xavier_uniform_(self.gating_network.weight)
        nn.init.constant_(self.gating_network.bias, 0.0)

    def forward(self, input_ids, attention_mask=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        gate = self.gating_network(pooled_output)

        if self.routing == 'soft':
            gating_weights = t.softmax(gate, dim=1)
            expert_outputs = t.stack([expert(pooled_output) for expert in self.experts], dim=1)
            output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'hard':
            gating_weights = t.softmax(gate, dim=1)
            top_k_weights, top_k_indices = t.topk(gating_weights, self.top_k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
            expert_outputs = t.stack([expert(pooled_output) for expert in self.experts], dim=1)
            sparse_weights = t.zeros(pooled_output.size(0), self.expert_number, device=gate.device)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            output = t.bmm(sparse_weights.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'gumbel':
            gating_weights = nn.functional.gumbel_softmax(gate, tau=self.tau, hard=self.gumbel_hard, dim=1)
            expert_outputs = t.stack([expert(pooled_output) for expert in self.experts], dim=1)
            output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)

        return output

    def router(self, input_ids, attention_mask=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        return self.gating_network(pooled_output)


class MoETransformerModel(nn.Module):
    """Lightweight Transformer with MoE classification head (MoE after encoder)"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, expert_number,
                 num_heads=4, num_layers=2, max_len=128, top_k=None,
                 routing='soft', tau=1.0, gumbel_hard=False, dropout=0.3):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.expert_number = expert_number
        self.routing = routing
        self.gumbel_hard = gumbel_hard
        self.tau = tau
        self.top_k = top_k if top_k is not None else expert_number

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.experts = nn.ModuleList([
            TextExpert2(embedding_dim, output_dim, dropout) for _ in range(expert_number)
        ])
        self.gating_network = nn.Linear(embedding_dim, expert_number)
        nn.init.xavier_uniform_(self.gating_network.weight)
        nn.init.constant_(self.gating_network.bias, 0.0)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def _encode(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        position_ids = t.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(position_ids))

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        encoded = self.layer_norm(encoded)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = encoded.mean(dim=1)

        return pooled

    def forward(self, input_ids, attention_mask=None):
        pooled = self._encode(input_ids, attention_mask)
        gate = self.gating_network(pooled)

        if self.routing == 'soft':
            gating_weights = t.softmax(gate, dim=1)
            expert_outputs = t.stack([expert(pooled) for expert in self.experts], dim=1)
            output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'hard':
            gating_weights = t.softmax(gate, dim=1)
            top_k_weights, top_k_indices = t.topk(gating_weights, self.top_k, dim=1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
            expert_outputs = t.stack([expert(pooled) for expert in self.experts], dim=1)
            sparse_weights = t.zeros(pooled.size(0), self.expert_number, device=gate.device)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            output = t.bmm(sparse_weights.unsqueeze(1), expert_outputs).squeeze(1)

        elif self.routing == 'gumbel':
            gating_weights = nn.functional.gumbel_softmax(gate, tau=self.tau, hard=self.gumbel_hard, dim=1)
            expert_outputs = t.stack([expert(pooled) for expert in self.experts], dim=1)
            output = t.bmm(gating_weights.unsqueeze(1), expert_outputs).squeeze(1)

        return output

    def router(self, input_ids, attention_mask=None):
        pooled = self._encode(input_ids, attention_mask)
        return self.gating_network(pooled)


class MoETransformerModelV2(nn.Module):
    """Transformer with MoE layers INSIDE each layer (like Switch Transformer)"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, expert_number,
                 num_heads=4, num_layers=2, max_len=128, top_k=2, routing='hard', dropout=0.3):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.num_layers = num_layers
        self.routing = routing
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MoETransformerLayer(
                d_model=embedding_dim,
                num_heads=num_heads,
                d_ff=hidden_dim,
                num_experts=expert_number,
                top_k=top_k,
                routing=routing,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, output_dim)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        position_ids = t.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(position_ids))

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        all_gate_probs = []
        for layer in self.layers:
            x, gate_probs = layer(x, src_key_padding_mask)
            all_gate_probs.append(gate_probs)

        x = self.final_norm(x)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        output = self.classifier(pooled)
        self.last_gate_probs = all_gate_probs

        return output

    def router(self, input_ids, attention_mask=None):
        """Get average routing weights across all layers and positions"""
        batch_size, seq_len = input_ids.shape

        position_ids = t.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        all_gate_probs = []
        for layer in self.layers:
            x, gate_probs = layer(x, src_key_padding_mask)
            # Average over sequence length: [batch_size, num_experts]
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                avg_gate = (gate_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                avg_gate = gate_probs.mean(dim=1)
            all_gate_probs.append(avg_gate)

        # Average across layers and return logits (inverse softmax approximation)
        avg_probs = t.stack(all_gate_probs, dim=0).mean(dim=0)

        return t.log(avg_probs + 1e-10)


class BertDenseBaseline(nn.Module):
    """BERT with simple dense classifier (no MoE) for comparison"""
    def __init__(self, pretrained_model_name, output_dim, hidden_multiplier=1, freeze_bert=False, dropout=0.3):
        super().__init__()
        self.bert = tf.BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * hidden_multiplier),
            nn.LayerNorm(self.hidden_size * hidden_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * hidden_multiplier, output_dim)
        )

    def forward(self, input_ids, attention_mask=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs.pooler_output
        return self.classifier(pooled_output)
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