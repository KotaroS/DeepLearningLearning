# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image
import random


def get_data(normalize ):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

x, t = get_data(True)
network = init_network()
accuracy_cnt = 0
#for i in range(len(x)):
#    y = predict(network, x[i])
#    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
#    if p == t[i]:
#        accuracy_cnt += 1

#print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

x2, t2 = get_data(False)
index = random.randint(0,10000)
img = x[index]
label = t[index]

y = predict(network, img)
print("正解ラベル：")
print(label)
print("ニューラルネットワークの推論結果：")
y2 = np.round(y*100)
for j in range(len(y2)):
  print(str(j) + "：" + str(y2[j]) + "%")


img2 = x2[index].reshape(28, 28)

img_show(img2)
