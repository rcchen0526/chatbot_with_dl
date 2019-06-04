import random
import jieba
import pickle

# custom dict
jieba.set_dictionary('jieba_dict/dict.txt.big')
# open raw data 
fp = open('data/PTT_data_filter.txt', "r")
# open new data
writer = open('data/new_PTT_data.txt', "w")
# 縮減為原本的1/num倍
num=8
# initial dict of label
word_list = set(' ')

line = fp.readline()
foo = [i for i in range(num)]
data_length=0
max_data_length=0
max_label_length=0
while line:   
    if 0==random.choice(foo):
        data, label = line.split('\t')
        data_words = list(jieba.cut(data, cut_all=False))
        for word in data_words[:-1]:
            writer.write(word+' ')
        writer.write(data_words[-1]+'\t')
        writer.write(label)
        data_length+=1
        # calculate the max length
        if len(data_words)>max_data_length:
            max_data_length = len(data_words)
        if len(label)>max_label_length:
            max_label_length = len(label)
        # make dict
        for i in range(len(label)):
            if label[i] not in word_list:
                word_list.add(label[i])
    line = fp.readline()
fp.close()
writer.close()
print('Total {} data'.format(data_length))
print('The max length of data is {}'.format(max_data_length))
print('The max length of label is {}'.format(max_label_length))

# make word2idx and idx2word
word2idx=dict()
idx2word=dict()
for i, item in enumerate(sorted(word_list)):
    word2idx[item]=i
    idx2word[i]=item
# save
with open('word2idx.dic', 'wb') as fp:
    pickle.dump(word2idx, fp)
with open('idx2word.dic', 'wb') as fp:
    pickle.dump(idx2word, fp)
'''
with open('idx2word_less.dic', 'rb') as fp:
    idx2word = pickle.load(fp)
print(idx2word)
'''
