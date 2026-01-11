import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

    def forward(self,x):
        expert_weight = self.gating_network(x)
        expert_out_list = [expert(x).unsqueeze(1) for expert in self.experts]
        expert_out = t.cat(expert_out_list,dim=1)
        expert_weight = t.softmax(expert_weight,dim=1).unsqueeze(1)
        output = expert_weight @ expert_out
        return output.squeeze()

