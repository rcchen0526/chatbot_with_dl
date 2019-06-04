import pickle
from argparse import ArgumentParser

with open('word2idx.dic', 'rb') as fp:
    word2idx = pickle.load(fp)
class_num = len(word2idx)
fp.close()

parser = ArgumentParser()
parser.add_argument("-lr", help="leraning rate", default=1e-3)
parser.add_argument("-bs", help="Batch_Size", default=128)
parser.add_argument("-nd", help="Length of Data", default=20)
parser.add_argument("-nl", help="Length of Label", default=28)
parser.add_argument("-class_num", help="Length of class", default=class_num)
parser.add_argument("-epoch", help="leraning rate", default=1000)
parser.add_argument("-workers", help="leraning rate", default=10)
parser.add_argument("-w2v_dim", help="Dim of word2vec", default=250)
parser.add_argument("-atten", help="Hidden layer of encoder", default=200)
parser.add_argument("-enc", help="Hidden layer of encoder", default=400)
parser.add_argument("-dec", help="Hidden layer of decoder", default=800)
parser.add_argument("-data_path", help="The path of training data", default='data/new_PTT_data.txt')
parser.add_argument("-word2idx_path", help="The path of word2idx", default='word2idx.dic')

args = parser.parse_args()
