from sklearn.svm import SVC
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm

def data_preprocess(split='train'):
    data_list = np.genfromtxt(f'dataset/{split}_list.txt', dtype=np.str, delimiter=',')
    X = []
    Y = []

    for filename, y in tqdm(data_list):
        y = int(y)

        if y == 0:
            x = Image.open(f'dataset/unmask/{filename}').convert('RGB')
        else:
            x = Image.open(f'dataset/mask/{filename}').convert('RGB')

        x = np.array(x.getdata()).reshape(-1)
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

X_train, y_train = data_preprocess('train')
X_test, y_test = data_preprocess('test')

# svc = SVC(kernel='linear')
# svc.fit(X_train, y_train)
#
# with open('svm.pkl', 'wb') as f:
#     pickle.dump(svc, f)

with open('svm.pkl', 'rb') as f:
    svc = pickle.load(f)

pred = svc.predict(X_test)

acc = (pred == y_test).mean()
print(acc)