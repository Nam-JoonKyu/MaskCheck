from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from augmix import AugMix


class MaskDataset(Dataset):

    def __init__(self, root, split, preprocess, no_aug=False):
        self.root = root
        self.split = split
        self.preprocess = preprocess
        self.no_aug = no_aug
        self.data_list = np.genfromtxt(os.path.join(root,f'{split}_list.txt'), dtype=str, delimiter=',')
        self.aug = AugMix()

    def __getitem__(self, idx):
        filename, y = self.data_list[idx]
        y = int(y)

        if y == 0:
            x = Image.open(os.path.join(self.root,'unmask',filename)).convert('RGB')
        else:
            x = Image.open(os.path.join(self.root,'mask', filename)).convert('RGB')

        if self.no_aug:
            return self.preprocess(x), y
        else:
            aug1 = self.aug.augment_and_mix(x, self.preprocess)
            aug2 = self.aug.augment_and_mix(x, self.preprocess)
            return (self.preprocess(x), aug1, aug2), y

    def __len__(self):
        return len(self.data_list)



