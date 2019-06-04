#coding=utf-8
import random
import pickle
import jieba
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from data_args import *
from data_load import *
from model import *

import warnings
warnings.filterwarnings("ignore")

# data generater
training_generator = DataGenerator(train_datas)

print('length of dict is {}'.format(class_num))

model = s2s_model(Tx=args.nd, Ty=args.nl, n_a=args.enc, n_s=args.dec, target_vocab_size=args.class_num)
#model = load_model('call_model.h5')
filepath = "call_model_atten.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.compile(optimizer=Adam(lr=args.lr), 
            loss='categorical_crossentropy')

model.fit_generator(training_generator, epochs=args.epoch, workers=args.workers, verbose=1, callbacks=callbacks_list)
#model.save('s2s_model.h5')
