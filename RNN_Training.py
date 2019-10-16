#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:15:33 2018

@author: david
"""

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking, Dropout, TimeDistributed
from keras.callbacks import History 

# In[1]:
## Data Preprocessing
# load data
origin_data = np.load('array_train_cup.npy')

# Random shuffle data
np.random.seed(7)
np.random.shuffle(origin_data)

# Extract the input data and groundtruth
data_x = np.delete(origin_data, 1, axis=2)
#data_x = origin_data[:,:,0]
#data_x = np.reshape(data_x,[origin_data.shape[0], origin_data.shape[1], 1])
data_y = origin_data[:,:,1]
data_y = np.reshape(data_y,[origin_data.shape[0], origin_data.shape[1], 1])
# Split data into train, test
train_size = 0.8
train_num = int(train_size*origin_data.shape[0])
train_x = data_x[0:train_num,:,:]
train_y = data_y[0:train_num,:,:]
test_x = data_x[train_num:,:,:]
test_y = data_y[train_num:,:,:]

# In[2]:
model_para = [2,100]
model_para = np.reshape(model_para,[1,2])
model_para = np.tile(model_para,(2,1))
ep_num = [100,120]
ep_num = np.reshape(ep_num,[2,1])
ep_num = np.repeat(ep_num,1,axis=0)
loop_para = np.concatenate((model_para,ep_num),axis=1)
loop_range = np.arange(loop_para.shape[0])

for iter_i in loop_range:
    
    ## Build the RNN model
    # parameters setting
    timesteps_num = data_x.shape[1]
    features_num = data_x.shape[2]
    out_size = 1
    LTSM_layers = loop_para[iter_i,0]
    LTSM_units = loop_para[iter_i,1]
    range(0,LTSM_layers)
    # create model
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps_num, features_num)))
    for num in range(0,LTSM_layers):
        model.add(LSTM(LTSM_units, return_sequences=True))
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(out_size,  activation='linear')))
 
    model.compile(loss='mse', optimizer='adam')

    model.summary()

    ## Train the MDN model
    fit_record=History()

    fit_record=model.fit(train_x, train_y, batch_size=32, epochs=loop_para[iter_i,2], 
                         validation_split=0.25)
    model.reset_states()
    model.save_weights('weights_'+str(iter_i)+'.h5')
    model.save('model_'+str(iter_i)+'.h5')

    train_loss=fit_record.history['loss']
    valid_loss=fit_record.history['val_loss']
    test_loss=model.evaluate(test_x, test_y)
    print(train_loss,valid_loss,test_loss)
    
    np.save('train_loss_'+str(iter_i), train_loss)
    np.save('valid_loss_'+str(iter_i), valid_loss)

    train_loss=fit_record.history['loss']
    valid_loss=fit_record.history['val_loss']

    pred=model.predict(test_x)

    plt.figure()
    si = 1
    #index = np.random.randint(test_x.shape[0],size=9)
    index = [18,36,56,102,211,185,136,147,89,45,222,250]
    for i in index:
        plt.subplot(4,3,si)
        x = test_y[i,:,0].nonzero()
        x = np.asarray(x)
        xs = np.arange(x.shape[1])
        plt.plot(xs, test_y[i,0:x.shape[1],0], 'r', xs, pred[i,0:x.shape[1],0], 'b--')
        si +=1
    plt.savefig("Figure_"+str(iter_i)+".png")
    plt.show()

















