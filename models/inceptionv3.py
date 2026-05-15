import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        b1   = self.branch1x1(x)
        b5   = self.branch5x5_2(self.branch5x5_1(x))
        b3db = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        bpool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b5, b3db, bpool], dim=1)

class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        b3   = self.branch3x3(x)
        b3db = self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x)))
        bpool = self.branch_pool(x)
        return torch.cat([b3, b3db, bpool], dim=1)

class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super().__init__()
        c7 = channels_7x7
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        b1   = self.branch1x1(x)
        b7   = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        b7db = self.branch7x7dbl_5(
                   self.branch7x7dbl_4(self.branch7x7dbl_3(
                   self.branch7x7dbl_2(self.branch7x7dbl_1(x)))))
        bpool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b7, b7db, bpool], dim=1)

class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

        self.branch_pool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        b3    = self.branch3x3_2(self.branch3x3_1(x))
        b7x3  = self.branch7x7x3_4(
                    self.branch7x7x3_3(self.branch7x7x3_2(self.branch7x7x3_1(x))))
        bpool = self.branch_pool(x)
        return torch.cat([b3, b7x3, bpool], dim=1)

class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        b1   = self.branch1x1(x)

        b3_t = self.branch3x3_1(x)
        b3   = torch.cat([self.branch3x3_2a(b3_t), self.branch3x3_2b(b3_t)], dim=1)

        b3db_t = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        b3db   = torch.cat([self.branch3x3dbl_3a(b3db_t),
                             self.branch3x3dbl_3b(b3db_t)], dim=1)

        bpool = self.branch_pool(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1))
        return torch.cat([b1, b3, b3db, bpool], dim=1)  # 320+768+768+192 = 2048

class Inception3(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        # Stem - in_channels=1 for grayscale (original had 3)
        self.conv1    = BasicConv2d(1, 32, kernel_size=3, stride=2)
        self.conv2    = BasicConv2d(32, 32, kernel_size=3)
        self.conv3    = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.conv4    = BasicConv2d(64, 80, kernel_size=1)
        self.conv5    = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        # Inception-A x3  
        self.inception_a = nn.Sequential(
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
        )

        # Inception-B x1  
        self.inception_b = InceptionB(288)

        # Inception-C x4  
        self.inception_c = nn.Sequential(
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192),
        )

        # Inception-D x1   
        self.inception_d = InceptionD(768)

        # Inception-E x2   
        self.inception_e = nn.Sequential(
            InceptionE(1280),
            InceptionE(2048),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        x = self.inception_a(x)
        x = self.inception_b(x)
        x = self.inception_c(x)
        x = self.inception_d(x)
        x = self.inception_e(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

