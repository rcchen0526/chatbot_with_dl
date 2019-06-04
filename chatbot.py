import random
import jieba
import pickle
import numpy as np
from keras.models import load_model
from data_args import *
from model import *

with open('idx2word.dic', 'rb') as fp:
    idx2word = pickle.load(fp)
with open('word2vec.dic', 'rb') as fp:
    vocab_list = pickle.load(fp)

jieba.set_dictionary('jieba_dict/dict.txt.big')
fp = open('data/testing.txt', "r")
line = fp.readline()
testing_vec=[]
datas=[]
while line:
    data, label = line.split('\t')
    datas.append(data)
    data_words = list(jieba.cut(data, cut_all=False))
    data_vec = np.zeros((args.nd, args.w2v_dim))
    for i, item in enumerate(data_words):
        try:
            _vec = vocab_list[item]
        except:
            _vec = np.random.normal(loc=0.0, scale=0.1, size=(args.w2v_dim))
        data_vec[i] = _vec
    data_vec = list(data_vec)
    testing_vec.append(data_vec)
    line = fp.readline()

testing_vec = np.array(testing_vec)

train_model = load_model('call_model_atten.h5')
model = test_model(Tx=args.nd, Ty=args.nl, n_a=args.enc, n_s=args.dec, target_vocab_size=args.class_num, train_model=train_model)

s0 = np.zeros((testing_vec.shape[0], args.dec))
c0 = np.zeros((testing_vec.shape[0], args.dec))
out0 = np.zeros((testing_vec.shape[0], args.class_num))

s, c = s0, c0
preds = model.predict([testing_vec, s0, c0, out0])

sentence_data=[]
sentence_label=[]
predictions = np.argmax(preds, axis=-1)

for text_data, text_label in zip(datas, list(predictions.swapaxes(0,1))):

    sentence_data.append(text_data) 
    print(text_data, end='\t')

    sentence=''
    for idx in text_label:
        word=idx2word[idx]
        if word == '\n' or idx==0:
            continue
        sentence+=idx2word[idx]
    sentence_label.append(sentence)
    print(sentence)