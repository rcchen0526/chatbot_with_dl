from keras.models import Model
from keras.layers import Input, Dense, RepeatVector, Concatenate, Dot, Reshape, Lambda
from keras.layers import Bidirectional,LSTM, GRU
from keras import backend as K
from keras import initializers
from data_args import *

repeator = RepeatVector(args.nd)
concatenator = Concatenate(axis=-1)

atten_dense = Dense(1, activation='softmax', kernel_initializer='random_normal', name='atten_dense')

dotor = Dot(axes = 1)
atten_GRU = GRU(args.atten, return_sequences=True, recurrent_initializer='orthogonal', name='GRU')

def GRU_attention(a, s_prev):
    global atten_GRU
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    energies = atten_GRU(concat)
    alphas = atten_dense(energies)
    context = dotor([alphas, a])
    return context

reshapor = Reshape((1, args.class_num))
concator = Concatenate(axis=-1)

decoder_LSTM_cell = LSTM(args.dec, return_state=True, recurrent_initializer='orthogonal', unit_forget_bias=True, name='decoder')
output_layer = Dense(args.class_num, activation='softmax', kernel_initializer='random_normal', name='output_dense')

def s2s_model(Tx, Ty, n_a, n_s, target_vocab_size):
    def slice(x,index):
        return x[:,index,:]
    # initial variable
    X = Input(shape=(Tx,args.w2v_dim))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    out0 = Input(shape=(Ty, target_vocab_size), name='out0')

    s = s0
    c = c0    
    outputs = []

    # Bi-LSTM encoder
    encoder = Bidirectional(LSTM(n_a, return_sequences=True, recurrent_initializer='orthogonal', unit_forget_bias=True), name='encoder')
    a = encoder(X)
    
    # Decoder
    for i in range(Ty):
        context = GRU_attention(a, s)
        # teacher forcing
        out = Lambda(slice,output_shape=(1,target_vocab_size),arguments={'index':i})(out0)

        context = concator([context, reshapor(out)])
        s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)
    model = Model([X, s0, c0, out0], outputs)
    
    return model

def test_attention(a, s_prev, atten, dense):

    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    energies = atten(concat)
    alphas = dense(energies)
    context = dotor([alphas, a])
    return context

def test_model(Tx, Ty, n_a, n_s, target_vocab_size, train_model):
    X = Input(shape=(Tx,args.w2v_dim))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    out0 = Input(shape=(target_vocab_size, ), name='out0')
    out = reshapor(out0)
    s = s0
    c = c0    
    outputs = []

    encoder = train_model.get_layer(name='encoder')
    atten_GRU = train_model.get_layer(name='GRU')
    atten_dense = train_model.get_layer(name='atten_dense')
    decoder_LSTM_cell = train_model.get_layer(name='decoder')
    output_layer = train_model.get_layer(name='output_dense')

    a = encoder(X)

    for i in range(Ty):
        context = test_attention(a, s, atten_GRU, atten_dense)
        # teacher forcing

        context = concator([context, reshapor(out)])
        s, _, c = decoder_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)
    model = Model([X, s0, c0, out0], outputs)
    
    return model