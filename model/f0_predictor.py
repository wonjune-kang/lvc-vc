import numpy as np
import torch
import torch.nn as nn


class F0PredictorNet(nn.Module):
    def __init__(self, embedding_dim):
        super(F0PredictorNet, self).__init__()

        self.layer_1 = nn.Linear(embedding_dim, 512)
        self.layer_out = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x
