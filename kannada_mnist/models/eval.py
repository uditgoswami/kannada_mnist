import torch

from kannada_mnist.data.data_loader import get_data_loaders
from kannada_mnist.models.utils import get_model
import pandas as pd


def load_model():
    model = get_model()
    model.load_state_dict(torch.load('models/kannada_mnist:22_0.83.pth'))
    model.eval()
    return model


def predict(model, dataloader):
    prediction_list = []
    with torch.no_grad():
        for img in dataloader:
            pred = model(img.cuda())
            pred = pred.cpu().detach().numpy().argmax(axis=1).tolist()
            prediction_list.extend(pred)
    return prediction_list


def main():
    model = load_model()
    _, test_dl, _, _ = get_data_loaders()
    preds = predict(model, test_dl)
    pred_df = pd.DataFrame({'id': range(5000), 'label': preds})
    pred_df.to_csv('data/submission.csv', index=False)
