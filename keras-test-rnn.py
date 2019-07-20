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


data = pd.read_csv("dummy2018.csv", usecols=[i for i in range(1, 80)], dtype=int)


# In[4]:


data.head()


# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l1, l2
from keras import backend


# In[6]:


model = Sequential()
model.add(LSTM(128, activation='tanh', kernel_initializer='glorot_uniform', input_shape=(256, 79)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(79, activation='sigmoid'))

opt = RMSprop(lr=0.001, clipnorm=1.0)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()


# In[7]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[8]:


data_gen = TimeseriesGenerator(data.values, data.values, length=256, stride=15, batch_size=16)


# In[9]:


x, y = data_gen[0]


# In[ ]:


history = model.fit_generator(data_gen, epochs=10, verbose=1)


# In[ ]:


model.save("test-lstm-1.h5")


# In[ ]:




