from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def resize(root='dataset/original_image', split='mask'):
    dir_path = os.path.join(root, split)
    i = 0
    for filename in tqdm(os.listdir(dir_path)):
        if filename[0] != '.':
            i+=1
            img = Image.open(os.path.join(dir_path,filename))
            img_resize = img.resize((128, 128), Image.BILINEAR)
            img_resize.save(os.path.join('dataset', split, f'{filename}'))

    print(f'{i} Images processed')

def train_test_split(root='./dataset', test_ratio=0.1):
    mask_list = [(f, 1) for f in os.listdir(os.path.join(root,'mask')) if f[0] != '.']
    mask_list = np.array(mask_list)

    indices = np.arange(len(mask_list))
    np.random.shuffle(indices)
    split = int(np.floor(len(mask_list) * test_ratio))
    train_indices, test_indices = indices[split:], indices[:split]

    mask_train_list = mask_list[train_indices]
    mask_test_list = mask_list[test_indices]

    unmask_list = [(f, 0) for f in os.listdir(os.path.join(root, 'unmask')) if f[0] != '.']
    unmask_list = np.array(unmask_list)

    indices = np.arange(len(unmask_list))
    np.random.shuffle(indices)
    split = int(np.floor(len(unmask_list) * test_ratio))
    train_indices, test_indices = indices[split:], indices[:split]

    unmask_train_list = unmask_list[train_indices]
    unmask_test_list = unmask_list[test_indices]

    train_list = np.concatenate((mask_train_list, unmask_train_list), axis=0)
    test_list = np.concatenate((mask_test_list, unmask_test_list), axis=0)

    np.random.shuffle(train_list)
    np.random.shuffle(test_list)

    with open('dataset/train_list.txt', 'w') as f:
        for filename, label in train_list:
            f.write(f'{filename},{label}\n')

    with open('dataset/test_list.txt', 'w') as f:
        for filename, label in test_list:
            f.write(f'{filename},{label}\n')



if __name__ == '__main__':
    #resize()
    #resize(split='unmask')
    train_test_split()