import sys
import argparse
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QAction, QFileDialog, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PIL import Image
from model import MaskNet
import pickle
import torch
import PyQt5.QtCore
import torchvision.transforms as transforms
import numpy as np

class MyApp(QMainWindow):

    def __init__(self, model_type='cnn'):
        super().__init__()
        self.initUI()

        assert model_type == 'cnn' or model_type == 'svm', "Choose one ['cnn', 'svm']"
        self.model_type = model_type

        if model_type == 'cnn':
            self.model = MaskNet([3, 64, 128, 256, 512, 512, 512])
            load_state = torch.load('checkpoint/masknet.ckpt', map_location='cpu')
            self.model.load_state_dict(load_state['model_state_dict'])
        else:
            with open('svm.pkl', 'rb') as f:
                self.model = pickle.load(f)



    def initUI(self):
        self.textEdit = QTextEdit()
        self.statusBar()
        self.qpixmap = QPixmap()
        self.lbl = QLabel(self)
        self.text = QLabel(self)
        self.text.setFont(QFont('Arial', 20))


        openFile = QAction(QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open New File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.setWindowTitle('Mask Check')
        self.setGeometry(300, 300, 600, 600)
        self.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File', './test_image')

        if fname[0]:
            self.qpixmap.load(fname[0])
            self.qpixmap.scaled(128, 128, PyQt5.QtCore.Qt.IgnoreAspectRatio)

            self.lbl.setPixmap(QPixmap(self.qpixmap))
            self.lbl.resize(self.qpixmap.width(), self.qpixmap.height())
            self.resize(self.qpixmap.width(), self.qpixmap.height()+100)

            self.text.setText('Predicting...')
            self.text.move(5, self.qpixmap.height()+30)

            x = Image.open(fname[0]).convert("RGB").resize((128, 128))

            if self.model_type == 'cnn':
                x = transforms.ToTensor()(x)
                x = x.unsqueeze(0)

                pred = self.model(x)
                pred = torch.argmax(pred, dim=-1).item()
            else:
                x = np.array(x.getdata()).reshape(1, -1)
                pred = self.model.predict(x)

            if pred == 0:
                self.text.setText('마스크를 착용해주세요.')
                self.text.setStyleSheet('Color:red')
            else:
                self.text.setText('통과해주세요.')
                self.text.setStyleSheet('Color:green')

            self.text.adjustSize()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet')

    parser.add_argument(
        '--model_type',
        type=str,
        default='cnn',
        choices=['cnn', 'svm']
    )

    args = parser.parse_args()
    app = QApplication(sys.argv)
    ex = MyApp(args.model_type)

    sys.exit(app.exec_())