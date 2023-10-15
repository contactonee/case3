import torch
from pathlib import Path
import cv2
import pandas as pd
import albumentations as A
import albumentations.core.serialization
from albumentations.pytorch import ToTensorV2
import torchvision

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.hub
import lightning as L
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import torch.mps
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchmetrics import F1Score
from sklearn.model_selection import train_test_split
import wandb
from functools import partial
import random

class TechosmotrDataset(Dataset):
    def __init__(self, images, transform=None, target_transform=None):

        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path, label = self.images[index]
        img = cv2.imread(img_path)

        if self.transform:
            if isinstance(self.transform,
                        albumentations.core.serialization.Serializable):
                img = self.transform(image=img)['image']
            else:
                img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        return img, label


if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    def scan_train_images(data_dir):
        data_dir = Path(data_dir)
        real_data_dir = data_dir / 'pravilniye(correct)'
        fake_data_dir = data_dir / 'fictivniye(fictitious)'

        real_images = real_data_dir.rglob('*.jpeg')
        fake_images = fake_data_dir.rglob('*.jpeg')

        data = []

        for img in real_images:
            data.append((img.as_posix(), 0))

        for img in fake_images:
            data.append((img.as_posix(), 1))

        return data


    def scan_test_images(data_dir):
        data_dir = Path(data_dir)

        images = data_dir.rglob('*.jpeg')

        data = []

        for img in images:
            data.append((img.as_posix(), 0))

        return data


    train_images = scan_train_images(
        'data/case3-datasaur-photo/techosmotr/techosmotr/train/')

    test_images = scan_test_images(
        'data/case3-datasaur-photo/techosmotr/techosmotr/test/')

    # train_images, val_images = train_test_split(train_images, test_size=1)

    val_images = random.sample(train_images, k=64)

    print(len(train_images), len(val_images))
        

    IMAGE_SIZE = 512

    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(),
        A.Affine(scale=[.9, 1.1], translate_percent=.2, rotate=[-180, 180], shear=0, p=1),
        # A.RandomSizedCrop([int(IMAGE_SIZE * 0.6), int(IMAGE_SIZE)], IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_test_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    target_transform = partial(torch.as_tensor, dtype=torch.long)


    train_ds = TechosmotrDataset(
        train_images,
        transform=train_transform,
        target_transform=target_transform
    )
    val_ds = TechosmotrDataset(
        val_images,
        transform=val_test_transform,
        target_transform=target_transform
    )
    test_ds = TechosmotrDataset(
        test_images,
        transform=val_test_transform,
        target_transform=target_transform
    )

    c1 = sum([label for _, label in train_images])
    c0 = len(train_images) - c1

    train_weights = [(c0+c1)/c0 if label == 0 else (c0+c1)/c0 for _, label in train_images]


    BATCH_SIZE = 16

    train_loader = DataLoader(
        train_ds,
        BATCH_SIZE,
        sampler=WeightedRandomSampler(train_weights, num_samples=max(c0, c1)*2),
        num_workers=2,
    )

    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2,)

    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2,)

    # model = torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 2, bias=True)

    model = torch.load('artifacts/oscar_epoch_20.pth')
        
    def train(dataloader, model, loss_fn, optimizer, epoch):
        total_loss = 0
        model.train()
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for batch, (X, y) in tqdm.tqdm(enumerate(dataloader)):
                X, y = X.to(device), y.to(device)

                y = nn.functional.one_hot(y, 2).to(torch.float)

                # Compute prediction error
                optimizer.zero_grad()
                pred = model(X).logits
                try:
                    loss = loss_fn(pred, y)
                except Exception as e:
                    print(X)
                    print(y)
                    print(X.shape)
                    print(y.shape)
                    print(e)

                total_loss += loss

                # Backpropagation
                loss.backward()
                optimizer.step()

                pbar.update()
                pbar.set_description(f'Epoch {epoch}\tLoss: {total_loss / len(dataloader):.7f}')
        
        return {'train_loss': total_loss / len(dataloader)}



    def validate(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for (X, y) in dataloader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                pred = logits.argmax(1)
                test_loss += loss_fn(logits, nn.functional.one_hot(y, 2).to(torch.float)).item()
                correct += (pred == y).type(torch.float).sum().item()
                

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Accuracy: {(100*correct):>0.1f}%\tTest loss: {test_loss:>8f} \n")
        
        return {
            'val_loss': test_loss,
            'val_acc': correct
        }
    torch.mps.empty_cache()

    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20

    wandb.init(
        # set the wandb project where this run will be logged
        project="datasaur-techosmotr",
        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-3,
            "architecture": str(type(model)),
            "epochs": epochs,
        }
    )

    for t in range(21, epochs+21):
        metrics = {}
        train_metrics = train(train_loader, model, loss_fn, optimizer, t)
        val_metrics = validate(val_loader, model, loss_fn)
        torch.save(model, f'artifacts/oscar_epoch_{t}.pth')
        
        metrics.update(train_metrics)
        metrics.update(val_metrics)
        wandb.log(metrics)

    wandb.finish()


    # model = torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 2, bias=True)

        
    # def train(dataloader, model, loss_fn, optimizer, epoch):
    #     total_loss = 0
    #     model.train()
    #     with tqdm.tqdm(total=len(dataloader)) as pbar:
    #         for batch, (X, y) in tqdm.tqdm(enumerate(dataloader)):
    #             X, y = X.to(device), y.to(device)

    #             y = nn.functional.one_hot(y, 2).to(torch.float)

    #             # Compute prediction error
    #             optimizer.zero_grad()
    #             pred = model(X).logits
    #             try:
    #                 loss = loss_fn(pred, y)
    #             except Exception as e:
    #                 print(X)
    #                 print(y)
    #                 print(X.shape)
    #                 print(y.shape)
    #                 print(e)

    #             total_loss += loss

    #             # Backpropagation
    #             loss.backward()
    #             optimizer.step()

    #             pbar.update()
    #             pbar.set_description(f'Epoch {epoch}\tLoss: {total_loss / len(dataloader):.7f}')
        
    #     return {'train_loss': total_loss / len(dataloader)}



    # def validate(dataloader, model, loss_fn):
    #     size = len(dataloader.dataset)
    #     num_batches = len(dataloader)
    #     model.eval()
    #     test_loss, correct = 0, 0
    #     with torch.no_grad():
    #         for (X, y) in dataloader:
    #             X, y = X.to(device), y.to(device)
    #             logits = model(X)
    #             pred = logits.argmax(1)
    #             test_loss += loss_fn(logits, nn.functional.one_hot(y, 2).to(torch.float)).item()
    #             correct += (pred == y).type(torch.float).sum().item()
                

    #     test_loss /= num_batches
    #     correct /= size
    #     print(
    #         f"Test Accuracy: {(100*correct):>0.1f}%\tTest loss: {test_loss:>8f} \n")
        
    #     return {
    #         'val_loss': test_loss,
    #         'val_acc': correct
    #     }
    # torch.mps.empty_cache()

    # model = model.to(device)
    # loss_fn = nn.BCEWithLogitsLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # epochs = 20

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="datasaur-techosmotr",
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": 1e-3,
    #         "architecture": str(type(model)),
    #         "epochs": epochs,
    #     }
    # )

    # for t in range(1, epochs+1):
    #     metrics = {}
    #     train_metrics = train(train_loader, model, loss_fn, optimizer, t)
    #     val_metrics = validate(val_loader, model, loss_fn)
    #     torch.save(model, f'artifacts/oscar_epoch_{t}.pth')
        
    #     metrics.update(train_metrics)
    #     metrics.update(val_metrics)
    #     wandb.log(metrics)

    # wandb.finish()