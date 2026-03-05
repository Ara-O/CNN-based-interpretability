# https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918

import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Softmax, CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.sgd import SGD
from torchinfo import summary
from torchvision.transforms import v2
from torch.optim import Adam
import pandas as pd
from sklearn.metrics import accuracy_score

data = pd.read_csv("../data/chest_xray/chest_xray_dataset.csv")
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

from PIL import Image
import os 

class XrayDataset(Dataset):
    def __init__(self, split, transform):
        self.data = data[data['split'] == split]

        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item = self.data.iloc[idx]
        label = self.data.iloc[idx]['class']
        image_data = Image.open(os.path.join("..", data_item['path']))
        image_data = self.transform(image_data)
        return image_data, label

train_transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.uint8, scale=True),
    v2.RandomResizedCrop(size=(500, 500)),
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.uint8, scale=True),
    v2.CenterCrop(size=(500, 500)),
    v2.ToDtype(torch.float32, scale=True),
])

train_dataset = XrayDataset(split='train', transform=train_transform)
val_dataset = XrayDataset(split="val", transform=test_transform)
test_dataset = XrayDataset(split="test", transform=test_transform)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64, num_workers=4)
val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=64, num_workers=4)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=64, num_workers=4)

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding='same')
        self.conv1_2 = Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding='same')
        self.pooling1 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2_1 = Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding='same')
        self.conv2_2 = Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='same')
        self.pooling2 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv3_1 = Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding='same')
        self.conv3_2 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same')
        self.conv3_3 = Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same')
        self.pooling3 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv4_1 = Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), padding='same')
        self.conv4_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding='same')
        self.conv4_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding='same')
        self.pooling4 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv5_1 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding='same')
        self.conv5_2 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding='same')
        self.conv5_3 = Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding='same')
        self.pooling5 = MaxPool2d(kernel_size=(2,2), stride=2)
        self.flatten = Flatten()
        self.fc1 = Linear(115200, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, 1)
        self.softmax = Softmax()
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pooling1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pooling2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pooling3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pooling4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pooling5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.softmax(x)
       
        return x

if __name__ == "__main__":
    model = VGG16().to(device)

    epochs = 15
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), 0.0001)

    for epoch in range(epochs):
        epoch_loss = 0
        print(f"Epoch {epoch}:")
        model.train()
        for idx, (x, target) in enumerate(train_dataloader):
            input = x.to(device)
            target = target.to(device)
            output = model(input)
            output = output.squeeze()
            loss = criterion(output, target.float())
            # zero the gradients
            optimizer.zero_grad()
            # calculate the gradients based on the calculated loss (the gradient of the loss wrt each param)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation loss
        print("Loss", epoch_loss/len(train_dataloader))
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_preds = []
            all_targets = []

            for x, target in val_dataloader:
                x = x.to(device)
                target = target.to(device).float()

                output = model(x).squeeze(-1)

                loss = criterion(output, target)
                val_loss += loss.item()

                preds = torch.sigmoid(output)
                all_preds.extend(preds.cpu())
                all_targets.extend(target.cpu())

            all_preds = torch.stack(all_preds)
            all_targets = torch.stack(all_targets)

            output_thresholded = (all_preds >= 0.5).float()
            print(output_thresholded)
            accuracy = accuracy_score(all_targets.numpy(), output_thresholded)

            print(f"Val loss: {val_loss / len(val_dataloader)} | Val accuracy: {accuracy}")


