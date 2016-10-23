#!/usr/bin/env python

# Small LSTM Network to Generate Text 
import sys
import pprint
import numpy
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load ascii text and covert to lowercase
filename = "smiths-clean.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique words to integers
raw_text_words = re.sub("[^\w]", " ",  raw_text).split()
words = sorted(list(set(raw_text_words)))
#pprint.pprint(words)
word_to_int = dict((w, i) for i, w in enumerate(words))
# summarize the loaded data
n_words = len(raw_text_words)
n_vocab = len(words)
print "Total Words: ", n_words
print "Total Vocab: ", n_vocab
# prepare the dataset of input to output pairs encoded as integers
seq_length = 10
dataX = []
dataY = []
for i in range(0, n_words - seq_length, 1):
	seq_in = raw_text_words[i:i + seq_length]
	seq_out = raw_text_words[i + seq_length]
	dataX.append([word_to_int[word] for word in seq_in])
	dataY.append(word_to_int[seq_out])
n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, nb_epoch=30, batch_size=128, callbacks=callbacks_list)

