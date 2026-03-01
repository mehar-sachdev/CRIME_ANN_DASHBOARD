import torch
import torch.nn as nn

class CrimeANN(nn.Module):
    def __init__(self, input_size, hidden_layers, neurons, dropout_rate=0.3):
        super(CrimeANN, self).__init__()

        layers = []
        in_features = input_size

        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, neurons))
            layers.append(nn.BatchNorm1d(neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = neurons

        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)