# For testing image must be reshaped to 
# Height = 32
# Width = 128
#


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import fnmatch
import cv2
import numpy as np
import string
import time



# char_list:   'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# total number of our output classes: len(char_list)
char_list = string.ascii_letters+string.digits

# Convolutional Recurrent Neural Network Architecture (CNN + RNN)

# input with shape of height=32 and width=128
inputs = Input(shape=(32, 128, 1))

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)

conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)

conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
# pooling layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)

conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)

conv_6 = Conv2D(512, (3, 3), activation='relu',
                padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)

conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(
    LSTM(128, return_sequences=True, dropout=0.2))(squeezed)
blstm_2 = Bidirectional(
    LSTM(128, return_sequences=True, dropout=0.2))(blstm_1)

outputs = Dense(len(char_list)+1, activation='softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)

act_model.load_weights('weights/crnn_model.h5')

img = cv2.cvtColor(cv2.imread("t1.jpg"), cv2.COLOR_BGR2GRAY)

w, h = img.shape

if w < 32:
	add_zeros = np.ones((32-w, h))*255
	img = np.concatenate((img, add_zeros))

if h < 128:
	add_zeros = np.ones((32, 128-h))*255
	img = np.concatenate((img, add_zeros), axis=1)
img = np.expand_dims(img , axis = 2)


img = img/255.0
valid_img = np.array(img)

valid_img = valid_img[np.newaxis, ...]


prediction = act_model.predict(valid_img)

out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
greedy=True)[0][0])

i = 0
for x in out:
	print("original_text = BASIC")
	print("predicted text = ", end = '')
	for p in x:
		if int(p) != -1:
			print(char_list[int(p)], end = '')
	i+=1
