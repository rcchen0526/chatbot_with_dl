import random
import jieba
import pickle
import numpy as np
from keras.models import load_model
from data_args import *
from model import *
import socket, sys
import json

with open('idx2word.dic', 'rb') as fp:
    idx2word = pickle.load(fp)
with open('word2vec.dic', 'rb') as fp:
    vocab_list = pickle.load(fp)

jieba.set_dictionary('jieba_dict/dict.txt.big')

train_model = load_model('call_model_atten.h5')
model = test_model(Tx=args.nd, Ty=args.nl, n_a=args.enc, n_s=args.dec, target_vocab_size=args.class_num, train_model=train_model)

class Server(object):
    def __init__(self, ip, port):
        global model
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #except socket.error, msg:
        except :
            sys.stderr.write("[ERROR] %s\n" % msg[1])
            sys.exit(1)
        HOST, PORT = ip, port
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #reuse tcp
        self.sock.bind((HOST, PORT))
        self.sock.listen(5)
    def response(self, msg):
        data_words = list(jieba.cut(msg, cut_all=False))
        data_vec = np.zeros((args.nd, args.w2v_dim))
        for i, item in enumerate(data_words):
            try:
                _vec = vocab_list[item]
            except:
                print('oov:{}'.format(item))
                _vec = np.random.normal(loc=0.0, scale=0.1, size=(args.w2v_dim))
            data_vec[i] = _vec
        data_vec = list(data_vec)
        testing_vec = []
        testing_vec.append(data_vec)
        testing_vec = np.array(testing_vec)
        s0 = np.zeros((testing_vec.shape[0], args.dec))
        c0 = np.zeros((testing_vec.shape[0], args.dec))
        out0 = np.zeros((testing_vec.shape[0], args.class_num))
        out0[:, 0] = 1
        s, c = s0, c0
        preds = model.predict([testing_vec, s0, c0, out0])

        predictions = np.argmax(preds, axis=-1)
        sentence=''
        preds = list(predictions.swapaxes(0,1))[0]
        for pred_idx in preds:
            word=idx2word[pred_idx]
            if word == '\n':
                break
            sentence+=idx2word[pred_idx]
        data = {'message': sentence}
        return data
    def run(self):
        print('Server Start!')
        while True:
            (csock, adr) = self.sock.accept()
            #print "Client Info: ", csock, adr
            msg = csock.recv(1024).decode()
            msg = msg[:-1]
            if not msg:
                pass
            else:
                print ("Client send: " + msg)
                try:
                    self.data = self.response(msg)
                except:
                    self.data = {'message':'Invalid Question'}
                print ('Response : {}'.format(self.data))
                self.data = json.dumps(self.data)
                csock.sendall(self.data.encode())
            csock.close()

def launch_server(ip, port):
    server = Server(str(ip), int(port))
    server.run()


if __name__ == '__main__':
    launch_server('0.0.0.0', 39391)
