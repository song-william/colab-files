import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import Sequential
from keras.layers import Dense, Dropout

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_file', type=str)
    parser.add_argument('--hidden_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    envname = args.policy_file
    filepath = 'expert_data/{}.pkl'.format(envname)
    x_train, y_train = load_data(filepath)

    train_bc(x_train, y_train, envname, hidden_layers=args.hidden_layers, epochs=args.epochs)


def train_bc(x_train, y_train, envname, hidden_layers=3, epochs=500):

    x_test = x_train[:-10]
    y_test = y_train[:-10]

    units = 64
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_dim=x_train.shape[1]))
    for _ in range(hidden_layers-1):
        model.add(Dense(units=units, activation='relu'))
    model.add(Dense(units=y_train.shape[1], activation='linear'))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=128)
    model.save("bc_policies/{}.h5".format(envname))
    return model


def load_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.loads(f.read())

    x_train = np.array(data['observations'])
    y_train = np.array(data['actions'])
    # reshape y
    x, _, y = y_train.shape
    y_train = np.reshape(y_train, (x, y))
    return x_train, y_train


if __name__ == '__main__':
    main()