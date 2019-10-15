import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from loguru import logger
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, dropout = 0.40):
        super(Net, self).__init__()
        self.dropout = dropout

        # https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch
        # Our batch shape for input x is (1, 28, 28)
        # (Batch, Number Channels, height, width).
        # Input channels = 1, output channels = 18
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=64)

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv1_1_bn = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.d2_1 = nn.Dropout2d(p=self.dropout)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.d2_2 = nn.Dropout2d(p=self.dropout)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.d2_3 = nn.Dropout2d(p=self.dropout)

        # 4608 input features, 256 output features (see sizing flow below)
        self.fc1 = nn.Linear(256 * 3 * 3, 512) # Linear 1
        self.d1_1 = nn.Dropout(p=self.dropout)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = nn.Linear(in_features=512, out_features=256) # linear 2
        self.d1_2 = nn.Dropout(p=self.dropout)
        self.fc3 = nn.Linear(in_features=256, out_features=128) # linear 3
        self.d1_3 = nn.Dropout(p=self.dropout)
        self.out = nn.Linear(in_features=128, out_features=10) # linear 3

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (1, 28, 28) to (18, 28, 28)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.conv1_1(x)
        x = self.conv1_1_bn(x)
        x = F.relu(x)

        x = self.d2_1(x)
        x = self.pool1(x) # Size changes from (18, 28, 28) to (18, 14, 14)

        # Second Conv
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.d2_2(x)
        x = self.pool2(x) # Size changes from (18, 14, 14) to (18, 7, 7)

        # Third Conv
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.d2_3(x)
        x = self.pool3(x) # Size changes from (18, 7, 7) to (18, 3, 3)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 14, 14) to (1, 3528)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 256 * 3 * 3)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = F.relu(self.fc1(x))
        x = self.d1_1(x)

        x = F.relu(self.fc2(x))
        x = self.d1_2(x)

        x = F.relu(self.fc3(x))
        x = self.d1_3(x)

        x = self.out(x)
        return F.log_softmax(x, dim=-1)


def get_model():
    model = Net()
    if cuda_present():
        model.cuda()
    return model


def cuda_present():
    return torch.cuda.is_available()


def get_num_correct(preds, labels):
    return (preds.argmax(axis=1) == labels).sum().item()


def train_classifier(model, dataloader, loss, optim, cuda):
    model.train()
    loss_tracker = 0
    correct_count = 0
    for imgs_, targets_ in tqdm(dataloader, "Batches"):
        if cuda:
            imgs_, targets_ = imgs_.cuda(), targets_.cuda()
        y_ = model(imgs_)
        losses = loss(y_, targets_)
        optim.zero_grad()
        losses.backward()
        optim.step()
        correct_count += get_num_correct(preds=y_.cpu().detach().numpy(), labels=targets_.cpu().detach().numpy())
        loss_tracker += (losses.cpu().detach().numpy())
    return loss_tracker/len(dataloader.dataset), correct_count/len(dataloader.dataset)


def val_classifier(model, dataloader, loss, cuda):
    model.eval()
    loss_tracker = 0
    correct_count = 0
    with torch.no_grad():
        for imgs_, targets_ in dataloader:
            if cuda:
                imgs_, targets_ = imgs_.cuda(), targets_.cuda()
            y_ = model(imgs_)
            losses = loss(y_, targets_)
            correct_count += get_num_correct(preds=y_.cpu().numpy(), labels=targets_.cpu().numpy())
            loss_tracker += (losses.cpu().detach().numpy())
    return loss_tracker/len(dataloader.dataset), correct_count/len(dataloader.dataset)


def checkpoint_model(*, model, label, metric: float, epoch: int):
    model_path = f"models/{label}:{epoch:0>2d}_{metric:.2f}.pth"
    torch.save(model.state_dict(), model_path)


def train(model, train_dl, val1_dl, val2_dl, loss, optim):
    records = []
    for i in tqdm(range(30), "Epochs"):
        _ = train_classifier(
            model=model,
            dataloader=train_dl,
            loss=loss,
            optim=optim,
            cuda=cuda_present(),
        )
        val1_loss, val1_score = val_classifier(
            model=model, dataloader=val1_dl, loss=loss, cuda=cuda_present()
        )
        val2_loss, val2_score = val_classifier(
            model=model, dataloader=val2_dl, loss=loss, cuda=cuda_present()
        )
        train_loss, train_score = val_classifier(
            model=model, dataloader=train_dl, loss=loss, cuda=cuda_present()
        )
        metrics = {
            "epoch": i + 1,
            "train_loss": train_loss,
            "val1_loss": val1_loss,
            "val2_loss": val2_loss,
            "train_score": train_score,
            "val1_score": val1_score,
            "val2_score": val2_score,
        }
        checkpoint_model(label='kannada_mnist', model=model, metric=val1_score, epoch=i + 1)
        records.append(metrics)
        print(f"\nEpoch: {i+1} \ntrain_loss: {train_loss}, train_score: {train_score} \n"
              f"val1_loss: {val1_loss}, val1_score: {val1_score} \n"
              f"val2_loss: {val2_loss}, val2_score: {val2_score}")
    return pd.DataFrame(records)
