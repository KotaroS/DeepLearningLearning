import pickle
import numpy as np
import os
import os.path
#from Train_data_create import initLinerData

keyFile = [
	'train_data',
	'train_label',
	'test_data',
	'test_label'
	]

dataset_dir = os.path.dirname(os.path.abspath(__file__))
def get_save_file_path(key):
	return '{0}/liner_{1}.pkl'.format(dataset_dir, key)

def initLinerData():
	print('called')

def _change_ont_hot_label(X):
	T = np.zeros((X.size, 10))
	for idx, row in enumerate(T):
			row[X[idx]] = 1

	return T

def load_liner(normalize=True, one_hot_label=False):
    """MNISTデータセットの読み込み
    
    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label : 
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    dataset = {}
    for key in keyFile:
        if not os.path.exists(get_save_file_path(key)):
            initLinerData()
        
        with open(get_save_file_path(key), 'rb') as f:
            dataset[key] = np.array(pickle.load(f))
    
    if normalize:
        for key in ('train_data', 'test_data'):
            mean = np.mean(dataset[key])
            std = np.std(dataset[key])
            dataset[key] = dataset[key].astype(np.float32)
            #dataset[key] /= 100.0
            dataset[key] = (dataset[key]-mean)/std
            
    if one_hot_label:
        dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])    

    return (dataset['train_data'], dataset['train_label']), (dataset['test_data'], dataset['test_label']) 


if __name__ == '__main__':
    initLinerData()
