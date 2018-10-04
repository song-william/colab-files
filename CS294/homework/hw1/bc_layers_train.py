import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
from keras.models import Sequential
from keras.layers import Dense, Dropout

# filename = 'Ant-v2'
filename = 'Humanoid-v2'
filepath = 'expert_data/{}.pkl'.format(filename)
with open(filepath, 'rb') as f:
        data = pickle.loads(f.read())

x_train = np.array(data['observations'])
y_train = np.array(data['actions'])
# reshape y
x, _, y = y_train.shape
y_train = np.reshape(y_train, (x, y))
print(x_train.shape)
print(y_train.shape)

for i in range(10):
    hidden_layers = i
    units = 64
    model = Sequential()
    model.add(Dense(units=units, activation='relu', input_dim=x_train.shape[1]))
    for _ in range(hidden_layers):
        model.add(Dense(units=units, activation='relu'))
    model.add(Dense(units=y_train.shape[1], activation='linear'))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mse'])
    model.fit(x_train, y_train, epochs=500, batch_size=128)
    model.save("experimental/{}_{}.h5".format(filename, hidden_layers))
