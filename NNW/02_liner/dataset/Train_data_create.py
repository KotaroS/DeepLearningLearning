import pickle
import numpy as np
import matplotlib.pyplot as plt

trainCount = 60000
testCount = 10000
dataSize = 10
outputClassCount = 10

keyFile = [
	'train_data',
	'train_label',
	'test_data',
	'test_label'
	]

def initLabel(data):
	tmp = np.zeros((data.shape[0], outputClassCount))
	maxIdx = data.argmax(axis = 1)
	for idx, row in enumerate(tmp):
		row[maxIdx[idx]] = 1
	return tmp.astype(np.int)

def initData(rCount, cCount):
	tmp = np.random.randint(0, 100,(rCount, cCount))
	return tmp

def initLinerData():
	dataset = {}
	# 訓練データ作成
	TRAIN_d = initData(trainCount, dataSize)
	# 訓練データ正解ラベル作成
	TRAIN_t = initLabel(TRAIN_d)

	# テストデータ作成
	TEST_d = initData(testCount, dataSize)
	# テストデータ正解ラベル作成
	TEST_t = initLabel(TEST_d)
	
	dataset[keyFile[0]] = TRAIN_d
	dataset[keyFile[1]] = TRAIN_t
	dataset[keyFile[2]] = TEST_d
	dataset[keyFile[3]] = TEST_t
	print(TRAIN_d.shape)
	print(TRAIN_t.shape)
	print(TEST_d.shape)
	print(TEST_t.shape)
	print(TRAIN_d)
	print(TRAIN_t)

	for key in keyFile:
		with open('liner_{0}.pkl'.format(key), 'wb') as f:
			pickle.dump(dataset[key].tolist(), f, -1)
	
	print('Finish')
	
	x = TRAIN_t.argmax(axis = 1)
	print(x.shape)
	plt.hist(x)
	plt.show()


if __name__ == '__main__':
    initLinerData()
