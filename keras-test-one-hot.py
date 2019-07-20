#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import h5py


# In[2]:


tf.test.is_gpu_available()


# In[3]:


data = pd.read_csv("onehot2018_5k.csv", usecols=[i for i in range(1, 163)], dtype=int)


# In[4]:


data.head()


# In[7]:


from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, BatchNormalization, CuDNNLSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l1, l2
from keras import backend


# In[15]:


model = Sequential()
# model.add(LSTM(162, activation='tanh', return_sequences=True, input_shape=(256, 162), kernel_regularizer = l2(0.001), activity_regularizer=l2(0.001)))
# model.add(LSTM(162, activation='tanh', return_sequences=True, kernel_regularizer = l2(0.001), activity_regularizer=l2(0.001)))
# model.add(LSTM(162, activation='tanh', kernel_regularizer = l2(0.001), activity_regularizer=l2(0.001)))

model.add(LSTM(162, return_sequences=True, input_shape=(512, 162)))
model.add(LSTM(162))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Dense(162, activation='sigmoid'))

opt = RMSprop(lr=0.0001, clipnorm=1.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()


# In[10]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[11]:


data_gen = TimeseriesGenerator(data.values, data.values, length=512, stride=32, batch_size=256, shuffle=True)


# In[ ]:


history = model.fit_generator(data_gen, epochs=5, verbose=1)


# In[ ]:


model.save("onehot2018_5k.h5")


# In[ ]:




