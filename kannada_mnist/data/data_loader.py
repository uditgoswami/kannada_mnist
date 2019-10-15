import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms


def load_data():
    train = pd.read_csv('data/train.csv')
    val = pd.read_csv('data/Dig-MNIST.csv')
    test = pd.read_csv('data/test.csv', index_col='id')
    return train, test, val


def get_transforms():
    train_trans = transforms.Compose(([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]))

    val_trans = transforms.Compose(([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]))
    return train_trans, val_trans


class KannadaDS(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        data = self.images[i].astype(np.uint8).reshape(28, 28, 1)

        if self.transforms:
            data = self.transforms(data)

        if self.labels is not None:
            return data, self.labels[i]
        else:
            return data


def get_data_loaders():
    train, test, val1 = load_data()
    val2 = train.sample(frac=0.2, random_state=7)
    train = train.drop(val2.index)
    train_trans, val_trans = get_transforms()
    train_data = KannadaDS(train.drop(columns=['label']).values, train.label.values, train_trans)
    val1_data = KannadaDS(val1.drop(columns=['label']).values, val1.label.values, val_trans)
    val2_data = KannadaDS(val2.drop(columns=['label']).values, val2.label.values, val_trans)
    test_data = KannadaDS(test.values, None, val_trans)
    train_dl = DataLoader(train_data, batch_size=512, shuffle=True, num_workers=6)
    val1_dl = DataLoader(val1_data, batch_size=512, shuffle=False, num_workers=6)
    val2_dl = DataLoader(val2_data, batch_size=512, shuffle=False, num_workers=6)
    test_dl = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=6)
    return train_dl, test_dl, val1_dl, val2_dl

