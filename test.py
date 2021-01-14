from model import MaskNet
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms

model = MaskNet([3, 64, 128, 256, 512, 512, 512])
load_state = torch.load('checkpoint/masknet_separable.ckpt', map_location='cpu')
model.load_state_dict(load_state['model_state_dict'])

X = []
for i in range(1, 16):
    filename = f'./test_image/{i}.png'
    x = Image.open(filename).convert('RGB').resize((128, 128))
    x = transforms.ToTensor()(x).unsqueeze(0)
    X.append(x)

x = torch.cat(X, dim=0)
y = [0, 1, 1, 0, 1,
     0, 0, 0, 0, 1,
     1, 1, 1, 0, 1]
pred = model(x)
pred = torch.argmax(pred, dim=-1)
pred = pred.tolist()

def metrics(y, pred):
    TP =