import torch.nn as nn
import torch.nn.functional as F

# CNN model for Sound Recognition
# class SoundCNN(nn.Module):
#     """CNN Model for ESC50 Dataset."""
#     def __init__(self):
#         super(SoundCNN, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(64 * 32 * 32, 128),
#             nn.ReLU(),
#             nn.Linear(128, 50)  # 50 classi per ESC-50
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc(x)
#         return x


import torch.nn as nn

class SoundCNN(nn.Module):
    """CNN Model for ESC50 Dataset with 3 Conv layers, BatchNorm, and Dropout after each block."""
    def __init__(self, dropout: float):
        super(SoundCNN, self).__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),  # 64x64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),  # 32x32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),  # 16x16
        )

        # Calcolo coerente:
        # Input: 128 -> 64 -> 32 -> 16
        # Canali finali: 128
        # Flatten: 128 * 16 * 16 = 32768
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 50)  # 50 classi per ESC-50
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    

import math

class SoundCNN_Variable(nn.Module):
    def __init__(self, input_size: int, kernel_size: int, stride: int, dropout: float, n_blocks: int):
        super(SoundCNN_Variable, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.padding = kernel_size // 2
        self.n_blocks = n_blocks

        in_channels = 1
        conv_blocks = []
        size = input_size

        for i in range(n_blocks):
            out_channels = 32 * (2 ** i)  # Es: 32, 64, 128, 256, ...
            conv_blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(dropout),
                nn.MaxPool2d(2)
            ]
            # Update size after conv and pooling
            size = self._conv_output_size(size)
            in_channels = out_channels

        self.conv = nn.Sequential(*conv_blocks)

        self.flatten_dim = out_channels * size * size  # out_channels aggiornato nell'ultimo blocco

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 50)
        )

    def _conv_output_size(self, size):
        # Calcolo dopo Conv2D con padding 'same'
        size = math.floor((size + 2 * self.padding - self.kernel_size) / self.stride + 1)
        size = size // 2  # MaxPool2d(2)
        return size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0, activation=F.silu):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=50, in_channels=1, init_filters=64, 
                 pool_size='adaptive', activation=F.silu, input_height=128, input_width=128, 
                 dropout=0.3):
        super(ResNet, self).__init__()
        self.in_planes = init_filters
        self.activation = activation
        self.pool_size = pool_size
        self.dropout = dropout

        self.conv1 = nn.Conv2d(in_channels, init_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_filters)

        self.layer1 = self._make_layer(block, init_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, init_filters * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, init_filters * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, init_filters * 8, num_blocks[3], stride=2)

        if pool_size == 'adaptive':
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            flattened_size = init_filters * 8 * block.expansion
        else:
            self.avg_pool = lambda x: F.avg_pool2d(x, pool_size)
            feature_height = input_height // 8
            feature_width = input_width // 8
            pooled_height = feature_height // pool_size if feature_height >= pool_size else 1
            pooled_width = feature_width // pool_size if feature_width >= pool_size else 1
            flattened_size = init_filters * 8 * block.expansion * pooled_height * pooled_width

        self.linear = nn.Linear(flattened_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout=self.dropout, activation=self.activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=50, input_width=128, input_height=128):
    return ResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        pool_size='adaptive',
        input_width=input_width,
        input_height=input_height,
        in_channels=1,  # mono audio spectrogram
        init_filters=64,
        activation=F.silu,
        dropout=0.3  # Tune if needed
    )