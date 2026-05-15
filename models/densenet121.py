import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm2d(num_input_features)
        self.conv1x1    = nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv3x3    = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.batchnorm1(x))
        out = F.relu(self.batchnorm2(self.conv1x1(out)))
        out = self.conv3x3(out)
        return torch.cat((x, out), dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        in_features = num_input_features
        for _ in range(num_layers):
            self.layers.append(DenseLayer(in_features, growth_rate))
            in_features += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features, compression=0.5):
        super().__init__()
        out = math.floor(num_input_features * compression)
        self.bn      = nn.BatchNorm2d(num_input_features)
        self.conv1x1 = nn.Conv2d(num_input_features, out, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.avgpool(self.conv1x1(F.relu(self.bn(x))))

class DenseNet121(nn.Module):
    def __init__(self, growth_rate: int = 32, dropout: float = 0.5):
        super().__init__()

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        dense_block_layers = [6, 12, 24, 16]
        channel_count = 64
        self.blocks = nn.ModuleList()

        for idx, num_layers in enumerate(dense_block_layers):
            block = DenseBlock(num_layers, channel_count, growth_rate)
            channel_count += growth_rate * num_layers
            self.blocks.append(block)

            if idx != len(dense_block_layers) - 1:
                transition = TransitionLayer(channel_count, compression=0.5)
                channel_count = math.floor(channel_count * 0.5)
                self.blocks.append(transition)

        self.bn_final = nn.BatchNorm2d(channel_count)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(channel_count, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
