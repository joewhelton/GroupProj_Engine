#! python3
# createHouseModel.py - import a new dataset and create a new house price model

import tensorflow as tf
import tesnsoflowjs
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
from keras.models import load_model
import os
import sys

def norm(x):
  return (x-train_stats['mean']) / train_stats['std']

def createHouseModel(filename):
    fn = sys.argv[1]
    
    if os.pathexists(fn):
        rawds = pd.read_csv(filename)
        def createModel(rawds)
    elif
        print('Error with file')
        
def createModel(rawds)
    #copy the data into a pandas dataframe
    ds = rawds.copy()
    #drop data that may not be useful
    ds = ds.drop([ds.bedrooms >10].index)
    ds = ds.drop(ds[ds.bedrooms == 0].index)
    ds = ds.drop(ds[ds.bathrooms == 0].index)
    
    #deal with date object column and change to numeric
    ds['sale_yr'] = pd.to_numeric(ds.date.str.slice(0, 4))
    ds['sale_month'] = pd.to_numeric(ds.date.str.slice(4, 6))
    ds['sale_day'] = pd.to_numeric(ds.date.str.slice(6, 8))
    
    #create new datafrace with 3 new colums for date
    ds = pd.DataFrame(ds, columns=['id',
        'sale_yr','sale_month','sale_day','price',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront', 'view',
        'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
        'zipcode','lat','long','sqft_living15','sqft_lot15',])
    #drop the id column
    ds = ds.drop(['id'], axis=1)
    
    #split into test and train
    train_ds = ds.sample(frac=0.8, random_state = 0)
    test_ds = ds.drop(train_ds.index)
    train_stats = train_ds.describe()
    train_stats.pop('price')
    train_stats = train_stats.transpose()
    
    #split off column to predict
    train_labels = train_ds.pop('price')
    test_labels = test_ds.pop('price')
    
    #normalize the data
    normed_train_ds = norm(train_ds)
    normed_test_ds = norm(test_ds)
    
    #use keras to build model
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_ds.keys())]))
    model.add(keras.layers.Dense(64, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer = tf.keras.optimizers.RMSprop(0.001), loss='mse', metrics=['mae', 'mse'])

    #train model
    EPOCHS = 200
    history = model.fit(
    normed_train_ds, train_labels,
    epochs = EPOCHS, validation_split = 0.2, verbose=0)
    print('Done')
    
    #save model as .h5 file
    model.save('my_model.h5') 
    
    #convert saved model to model.json and binary weights file. 
    mkdir tfjs_files
    tensorflowjs_converter --input_format keras 'my_model.h5' 'tfjs_files'

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    housing_tflite_model = converter.convert()
    # Re-convert the model to TF Lite using quantization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    
    f = open('fullHsePrice.tflite', "wb")
    f.write(tflite_quantized_model)
    f.close()
    
createHouseModel('myFile')