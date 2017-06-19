#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import h5py
import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam

parser = argparse.ArgumentParser()

# yapf: disable
parser.add_argument('-x', '--observations', type=str,  default='obs.h5')
parser.add_argument('-y', '--actions',      type=str,  default='act.h5')
parser.add_argument('-n', '--new',          type=bool, default=False)
parser.add_argument('-d', '--depth',        type=int,  default=2)
parser.add_argument('-v', '--verbose',      type=bool, default=True)
parser.add_argument('-s', '--summary',      type=bool, default=False)
parser.add_argument('-e', '--epochs',       type=int,  default=1)
parser.add_argument('-o', '--output',       type=str,  default='my_model.h5')
parser.add_argument('-m', '--model',        type=str,  default='my_model.h5')
# yapf: enable

args = parser.parse_args()

N = 128
BATCH_SIZE = 128
BATCH_SIZE = 1
NUM_EPOCHS = args.epochs
VERBOSE = args.verbose
VALIDATION_SPLIT = 0.3
OPTIMIZER = Adam()
weights_filepath = args.model

CALLBACKS = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
    keras.callbacks.ModelCheckpoint(
        weights_filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)]

with h5py.File(args.observations, 'r') as hf:
    x = hf['obs'][:]
    n_x = x.shape[-1]  # 111
with h5py.File(args.actions, 'r') as hf:
    y = hf['act']
    n_y = y.shape[-1]  # 8
    y = y.reshape(-1, n_y)


def create_model():
    model = Sequential()

    model.add(Dense(N, activation='relu', input_shape=(n_x, )))

    for _ in range(args.depth):
        model.add(Dense(N, activation='relu'))

    model.add(Dense(n_y, activation='linear'))

    model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])
    return model


if args.new:
    model = create_model()
elif os.path.exists(args.model):
    model = load_model(args.model)
else:
    print(
        'No model trained. Creating default model with depth' +
        str(args.depth) + '.')
    model = create_model()

if args.summary:
    print(model.summary())

if NUM_EPOCHS == 0:
    pass
elif NUM_EPOCHS < 0:
    print('Need >= 0 epochs.')
else:
    model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=VERBOSE,
        validation_split=VALIDATION_SPLIT,
        callbacks=CALLBACKS, )
