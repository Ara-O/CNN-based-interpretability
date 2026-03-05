# %% [markdown]
# https://karan3-zoh.medium.com/paper-summary-imagenet-classification-with-deep-convolutional-neural-networks-41ce6c65960

# %%
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torch.optim import Adam
from sklearn.metrics import accuracy_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(11,11), stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=6400, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(4096, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(3,3), stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(3,3), stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=(3,3), stride=2)
        x = self.flatten(x)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from PIL import Image
import os 

class XrayDataset(Dataset):
    def __init__(self, split, transform):
        data = pd.read_csv("../data/chest_xray/chest_xray_dataset.csv")
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


if __name__ == "__main__":
    model = AlexNet()
    model.to(device)

    # t
    raw_transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(227, 227)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    raw_dataset = XrayDataset(split='train', transform=raw_transform)
    loader = DataLoader(raw_dataset, batch_size=64, num_workers=4)

    mean = 0.0
    std = 0.0
    for images, _ in loader:
        mean += images.mean()
        std += images.std()

    mean /= len(loader)
    std /= len(loader)

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.uint8, scale=True),
        v2.CenterCrop(256),
        v2.RandomCrop(225),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std])
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Grayscale(num_output_channels=1),
        v2.ToDtype(torch.uint8, scale=True),
        v2.CenterCrop(225),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[mean], std=[std])
    ])

    train_dataset = XrayDataset(split='train', transform=train_transform)
    val_dataset = XrayDataset(split="val", transform=test_transform)
    test_dataset = XrayDataset(split="test", transform=test_transform)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=64, num_workers=6)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=64, num_workers=6)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=64, num_workers=6)
 
    from tqdm import tqdm 

    epochs = 15
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), 0.0001)

    for epoch in  range(epochs):
        epoch_loss = 0
        print(f"Epoch {epoch}:")
        model.train()
        for idx, (x, target) in tqdm(enumerate(train_dataloader)):
            input = x.to(device)
            optimizer.zero_grad()
            target = target.to(device)
            output = model(input)
            output = output.squeeze(1)
            loss = criterion(output, target.float())
            # zero the gradients
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

                output = model(x).squeeze()

                loss = criterion(output, target)
                val_loss += loss.item()

                preds = torch.sigmoid(output)
                all_preds.extend(preds.cpu())
                all_targets.extend(target.cpu())

            all_preds = torch.stack(all_preds)
            all_targets = torch.stack(all_targets)

            output_thresholded = (all_preds >= 0.5).float()
            print(output_thresholded, all_targets)
            accuracy = accuracy_score(all_targets.numpy(), output_thresholded)

            print(f"Val loss: {val_loss / len(val_dataloader)} | Val accuracy: {accuracy}")

 
    model.eval()
    with torch.no_grad():
        test_loss = 0
        all_preds = []
        all_targets = []

        for x, target in test_dataloader:
            x = x.to(device)
            target = target.to(device).float()

            output = model(x).squeeze()

            loss = criterion(output, target)
            test_loss += loss.item()

            preds = torch.sigmoid(output)
            all_preds.extend(preds.cpu())
            all_targets.extend(target.cpu())

        all_preds = torch.stack(all_preds)
        all_targets = torch.stack(all_targets)

        output_thresholded = (all_preds >= 0.5).numpy()
        accuracy = accuracy_score(all_targets.numpy(), output_thresholded)

        print(f"Test loss: {test_loss / len(test_dataloader)} | Test accuracy: {accuracy}")
