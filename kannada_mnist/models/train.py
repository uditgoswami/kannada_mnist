import torch
from torch.nn import CrossEntropyLoss

from kannada_mnist.data.data_loader import get_data_loaders
from kannada_mnist.models.utils import train, get_model


def main():
    net = get_model()
    optim = torch.optim.Adam(net.parameters(), lr=4e-3)
    loss = CrossEntropyLoss()
    train_dl, test_dl, val1_dl, val2_dl = get_data_loaders()
    if torch.cuda.is_available():
        net = net.cuda()

    train(model=net, train_dl=train_dl, val1_dl=val1_dl, val2_dl=val2_dl, loss=loss, optim=optim)
