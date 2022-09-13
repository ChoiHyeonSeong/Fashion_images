import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# data = pd.read_csv('C:/Users/huns1/OneDrive/바탕 화면/Pycharm/Dataset/Image_1/train.csv')
data = pd.read_csv('C:/Users/Chs/Desktop/Fashion_images/data/train.csv')
train = data.iloc[:,2:]
label = data.iloc[:,1:2]

train = np.array(train,dtype='uint8')
label = np.array(label)
train.shape

train_reshape = train.reshape(60000, 28, 28)
train_reshape.shape
train_reshape

ok_lab = [0, 1, 2, 3, 4, 6, 7, 8, 9]
i=1	# 파일명을 다르게 지정하기 위한 변수
for img, lab in zip(train_reshape, label):
  if lab in ok_lab:
    # 256x256 resize, 보간법 적용
    img_resize = cv2.resize(img, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
    # 선명하게 만들기
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    img_sharp = cv2.filter2D(img_resize, -1, kernel)
    # 이미지 파일 생성
    cv2.imwrite('data/'+str(lab) + '_' + str(i)+'.jpg', img_sharp)	# 파일명 ex : 0_1.jpg
  i += 1