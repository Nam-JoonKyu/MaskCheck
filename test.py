from model import MaskNet
import torch
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def metrics(y, pred):
    TP = ((y == pred) & y).sum()
    TN = ((y == pred) & np.logical_not(y)).sum()
    FP = ((y != pred) & np.logical_not(y)).sum()
    FN = ((y != pred) & y).sum()

    acc = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    return acc, recall, precision


if __name__ == '__main__':

    model = MaskNet([3, 64, 128, 256, 512, 512, 512])
    load_state = torch.load('checkpoint/masknet_separable.ckpt', map_location='cpu')
    model.load_state_dict(load_state['model_state_dict'])

    X = []
    for i in range(1, 14):
        filename = f'./test_image/{i}.png'
        x = Image.open(filename).convert('RGB').resize((128, 128))
        x = transforms.ToTensor()(x).unsqueeze(0)
        X.append(x)

    x = torch.cat(X, dim=0)
    y=np.array([0, 1, 1, 0, 1,
                0, 0, 0, 0, 1,
                1, 1, 1])
    pred = model(x)
    pred = torch.argmax(pred, dim=-1)
    pred = np.array(pred)

    acc, recall, precision = metrics(y, pred)
    print(acc, recall, precision)




