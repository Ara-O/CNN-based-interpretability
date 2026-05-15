import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VGG16(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()

        def conv_block(*layer_configs):
            layers = []
            for in_c, out_c in layer_configs:
                layers += [
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block((1, 64),   (64, 64)),           # block 1
            conv_block((64, 128), (128, 128)),          # block 2
            conv_block((128, 256),(256, 256),(256,256)),# block 3
            conv_block((256, 512),(512, 512),(512,512)),# block 4
            conv_block((512, 512),(512, 512),(512,512)),# block 5
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
