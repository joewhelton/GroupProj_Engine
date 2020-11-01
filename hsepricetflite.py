# -*- coding: utf-8 -*-
"""HsePriceTFLite.ipynb

Created by Helen Daly R00142752
Original file is located at
    https://colab.research.google.com/drive/1vCyLJah-ZPAeSki3vbGJuzafvM1kyyIl
    I created this on colab when having technical difficulties installing tensorflow
    on my laptop. 
"""

!pip install tensorflow

"""### #import tensorflow"""

import tensorflow as tf
print(tf.__version__)

"""## #import all other packages"""

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

"""### #read in data and copy it to manipulate"""

rawds=pd.read_csv('/content/kc_house_data.csv')
rawds.head()

ds = rawds.copy()
ds.tail

"""### #visualise data"""

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
# %matplotlib inline
fig = plt.figure(figsize=(10,7))
fig.add_subplot(2,1,1)
sns.distplot(ds['price'])
fig.add_subplot(2,1,2)
sns.boxplot(ds['price'])
plt.tight_layout()

fig = plt.figure(figsize=(10,7))
fig.add_subplot(2,1,1)
sns.distplot(ds['bedrooms'])
fig.add_subplot(2,1,2)
sns.boxplot(ds['bedrooms'])
plt.tight_layout()

fig = plt.figure(figsize=(10,7))
fig.add_subplot(2,1,1)
sns.distplot(ds['bathrooms'])
fig.add_subplot(2,1,2)
sns.boxplot(ds['bathrooms'])
plt.tight_layout()

fig = plt.figure(figsize=(10,7))
fig.add_subplot(2,1,1)
sns.distplot(ds['sqft_living'])
fig.add_subplot(2,1,2)
sns.boxplot(ds['sqft_living'])
plt.tight_layout()

"""### #look at outlier data and drop if not correct"""

ds.loc[ds['bathrooms'] >= 7]

ds.info

ds.loc[ds['bedrooms']>10]

"""### #drop if in error"""

ds = ds.drop(ds[ds.bedrooms > 10].index)

ds.info

ds = ds.drop([8546])

ds.loc[ds['bathrooms'] >= 7]

ds.info

ds.loc[ds['price']>6000000]

ds.loc[ds['sqft_living']> 10000]

ds.loc[ds['sqft_living']< 20]

ds.info()

ds.isnull().sum()

ds.describe()

ds.loc[ds['bedrooms']== 0]

ds = ds.drop(ds[ds.bedrooms == 0].index)

ds.loc[ds['bathrooms']==0]

ds = ds.drop(ds[ds.bathrooms == 0].index)

ds.info

ds.describe

ds = ds.drop(['id', 'date', 'bathrooms', 'floors', 'condition', 'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long', 'sqft_living15'], axis=1)

ds.dtypes

ds['sale_yr'] = pd.to_numeric(ds.date.str.slice(0, 4))
ds['sale_month'] = pd.to_numeric(ds.date.str.slice(4, 6))
ds['sale_day'] = pd.to_numeric(ds.date.str.slice(6, 8))

ds = pd.DataFrame(ds, columns=[
        'sale_yr','sale_month','sale_day',
        'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
        'condition','grade','sqft_above','sqft_basement','yr_built',
        'zipcode','lat','long','sqft_living15','sqft_lot15','price'])

ds = ds.drop(['waterfront', 'view', 'yr_renovated'], axis=1)

ds.dtypes

"""### #split into test and train"""

train_ds = ds.sample(frac=0.8, random_state = 0)
test_ds = ds.drop(train_ds.index)

train_ds.info

"""### #get some statistics to compare later"""

train_stats = train_ds.describe()
train_stats.pop('price')
train_stats = train_stats.transpose()
train_stats

"""### #split off column to predict"""

train_labels = train_ds.pop('price')
test_labels = test_ds.pop('price')



"""### #normalize the data"""

def norm(x):
  return (x-train_stats['mean']) / train_stats['std']
normed_train_ds = norm(train_ds)
normed_test_ds = norm(test_ds)

"""### #use Keras to build model. We used RMSprop, as results better than adam"""

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_ds.keys())]))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(1))
model.compile(optimizer = tf.keras.optimizers.RMSprop(0.001), loss='mse', metrics=['mae', 'mse'])

model.summary()

"""### #see if model works"""

example_batch = normed_train_ds[:10]
example_result = model.predict(example_batch)
example_result

"""### #train data"""

EPOCHS = 500
history = model.fit(
    normed_train_ds, train_labels,
    epochs = EPOCHS, validation_split = 0.2, verbose=0)

"""### #evaluate model."""

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('MAE price [price]')
  plt.plot(hist['epoch'], hist['mae'],
           label = 'Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'VAL Error')
  plt.legend()
  plt.ylim([0,200000])

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('MSE [price]')
  plt.plot(hist['epoch'], hist['mse'],
           label = 'Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'VAL Error')
  plt.legend()
  plt.ylim([0,5000000000])

plot_history(history)

loss,mae,mse = model.evaluate(normed_test_ds, test_labels, verbose=0)
print('Testing set MAE: {:5.2f} price'.format(mae))

"""### #save the model"""

!mkdir -p FullHsPr_model
model.save('FullHsPr_model/my_model')

"""### #convert saved model to tensorflow lite for use with android and check size and quantize to make as small as possible."""

converter = tf.lite.TFLiteConverter.from_keras_model(model)
housing_tflite_model = converter.convert()

float_model_size = len(housing_tflite_model) / 1024
print('Float model size = %dKBs.' % float_model_size)

# Commented out IPython magic to ensure Python compatibility.
# Re-convert the model to TF Lite using quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Show model size in KBs.
quantized_model_size = len(tflite_quantized_model) / 1024
print('Quantized model size = %dKBs,' % quantized_model_size)
print('which is about %d%% of the float model size.'\
#       % (quantized_model_size * 100 / float_model_size))

"""### #download new model"""

f = open('fullHsePrice.tflite', "wb")
f.write(tflite_quantized_model)
f.close()

# Download the digit classification model
from google.colab import files
files.download('fullHsePrice.tflite')

"""### #examine shape of model"""

interpreter = tf.lite.Interpreter(model_path='/content/fullHsePrice.tflite')
interpreter.allocate_tensors()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])  # Example: [1 224 224 3]
print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

# Print output shape and type
print(interpreter.get_output_details()[0]['shape'])  # Example: [1 1000]
print(interpreter.get_output_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

"""#install tensorflowjs to convert for web ui"""

!pip install tensorflowjs

"""#to convert from saved model"""

import time
t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)

!tensorflowjs_converter --input_format=keras /content/1604250002.h5 /content/tfjsHsPrice_model