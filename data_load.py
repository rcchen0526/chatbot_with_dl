import math
import numpy as np
import pickle
import keras
from keras.models import Sequential
from data_args import *

#w2v_model = models.Word2Vec.load('word2vec.model')
with open(args.word2idx_path, 'rb') as fp:
    word2idx = pickle.load(fp)
with open('word2vec.dic', 'rb') as fp:
    vocab_list = pickle.load(fp)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, datas, shuffle=True):
        self.batch_size = args.bs
        self.data_length = args.nd
        self.label_length = args.nl
        self.w2v_dim = args.w2v_dim
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        # generate batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_datas = [self.datas[k] for k in batch_indexs]
        x, y = self.data_generation(batch_datas)
        return x, y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):

        datas = []
        labels = np.array([])
        teacher_labels = np.array([])

        # generate data
        for i, data in enumerate(batch_datas):
            #x_train data
            data_list = data['data']
            _data = np.zeros((self.data_length, self.w2v_dim))
            label = np.zeros((self.label_length, args.class_num))
            teacher_label = np.zeros((self.label_length, args.class_num))
            for i, item in enumerate(data_list):
                try:
                    _vec = vocab_list[item]
                except:
                    _vec = np.random.normal(loc=0.0, scale=0.1, size=(self.w2v_dim))
                _data[i] = _vec
            _data = list(_data)
            datas.append(_data)
            #y_train data
            label = np.zeros((self.label_length, args.class_num))
            idx_list = data['label']
            teacher_idx = idx_list.copy()
            teacher_idx.insert(0, 0)

            idx_list = np.array(idx_list)
            label[np.arange(len(idx_list)), idx_list] = 1
            teacher_label[np.arange(len(teacher_idx)), teacher_idx] = 1
            teacher_label = teacher_label[np.newaxis, :]
            label = label[np.newaxis, :].swapaxes(0,1)
            try:
                labels = np.concatenate((labels, label), axis=1)
            except:
                labels=label
            try:
                teacher_labels = np.concatenate((teacher_labels, teacher_label), axis=0)
            except:
                teacher_labels=teacher_label
        s0 = np.zeros((len(datas), args.dec))
        c0 = np.zeros((len(datas), args.dec))
        return [np.array(datas), s0, c0, teacher_labels], list(labels)

# read data
fp = open(args.data_path, "r")

train_datas = []
line = fp.readline()
while line:
    data, label = line.split('\t')
    data_words = data.split()
    label_words = []
    for i in range(len(label)):
        label_words.append(word2idx[label[i]])
    data_dict={'data':data_words, 'label':label_words}
    train_datas.append(data_dict)
    line = fp.readline()
fp.close()