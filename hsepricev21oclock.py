# -*- coding: utf-8 -*-
"""HsePriceV21oclock

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TR6y6aLdc-MEFT1tFR_Udr-zfOil4ymA
"""

!pip install tensorflow

import tensorflow as tf
print(tf.__version__)

"""### #import necessary libraries"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model

##read in data
from google.colab import files
file = files.upload()
ds=pd.read_csv('kc_house_data.csv')
ds.head()
#ds = keras.utils.get_file("kc_house_data.csv", "https://www.kaggle.com/harlfoxem/housesalesprediction?select=kc_house_data.csv")

ds.info()

"""#check if any null values"""

ds.isnull().sum()

ds.describe()

"""### #have a look at the datatypes"""

ds['sale_yr'] = pd.to_numeric(ds.date.str.slice(0, 4))
ds['sale_month'] = pd.to_numeric(ds.date.str.slice(4, 6))
ds['sale_day'] = pd.to_numeric(ds.date.str.slice(6, 8))

ds = pd.DataFrame(ds, columns=[
        'sale_yr','sale_month','sale_day',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'condition','grade','sqft_above','sqft_basement','yr_built',
        'zipcode','lat','long','sqft_living15','sqft_lot15','price'])
label_col = 'price'

print(ds.describe())

"""### #split into test and train"""

train_ds = ds.sample(frac=0.8, random_state = 0)
test_ds = ds.drop(train_ds.index)

"""### #look at overall statistics"""

train_stats= train_ds.describe()
train_stats.pop('price')
train_stats = train_stats.transpose()
train_stats

"""### #split off column to predict"""

train_labels = train_ds.pop('price')
test_lables = test_ds.pop('price')

"""### #normalize the data"""

def norm(x):
  return (x-train_stats['mean']) / train_stats['std']
normed_train_ds = norm(train_ds)
normed_test_ds = norm(test_ds)

"""### #use Keras to build model"""

model = keras.Sequential()
  model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_ds.keys())]))
  model.add(keras.layers.Dense(64, activation=tf.nn.relu))
  model.add(keras.layers.Dense(1))

  model.compile(optimizer = tf.keras.optimizers.RMSprop(0.001), loss='mse',
                metrics=['mae', 'mse'])

"""### #summary"""

model.summary()

"""### #try model"""

example_batch = normed_train_ds[:10]
example_result = model.predict(example_batch)
example_result

"""### #train the data"""

EPOCHS = 1000
history = model.fit(
    normed_train_ds, train_labels,
    epochs = EPOCHS, validation_split = 0.2, verbose=0)

"""### #evalutate the model"""

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()