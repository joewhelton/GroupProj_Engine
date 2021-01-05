#!/usr/bin/env python
# coding: utf-8

# In[312]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[313]:


# importing librabries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import tensorflow as tf
from tensorflow import keras

import keras
#from keras_vggface.vggface import VGGFace
#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))


# %%
#Basic and most important libraries
import pandas as pd , numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly

#Classifiers
from sklearn.ensemble import AdaBoostClassifier , GradientBoostingClassifier , VotingClassifier , RandomForestClassifier
from sklearn.linear_model import LogisticRegression , RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.svm import SVC

#Model evaluation tools
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import cross_val_score

# Import pickle Package
import pickle

#Data processing functions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[314]:


# Loading data

df_train = pd.read_csv('C:\\CIT\\GROUP\\ML\\LoanPredictionData\\train.csv')
df_test = pd.read_csv('C:\\CIT\\GROUP\\ML\\LoanPredictionData\\test.csv')


# In[315]:


# making missing value data frame

missing_df_train = df_train.isnull()
missing_df_test = df_test.isnull()


# In[316]:


# finding all missing values in training data

for column in missing_df_train.columns.values.tolist():
    print(column)
    print(missing_df_train[column].value_counts())
    print()


# In[317]:


# %%

print(df_train["Gender"].value_counts())
print(df_train["Married"].value_counts())
print(df_train["Self_Employed"].value_counts())
print(df_train["Dependents"].value_counts())
print(df_train["Credit_History"].value_counts())
print(df_train["Loan_Amount_Term"].value_counts())


# In[318]:


# %%
#Filling all Nan values with mode of respective variable
df_train["Gender"].fillna(df_train["Gender"].mode()[0],inplace=True)
df_train["Married"].fillna(df_train["Married"].mode()[0],inplace=True)
df_train["Self_Employed"].fillna(df_train["Self_Employed"].mode()[0],inplace=True)
df_train["Loan_Amount_Term"].fillna(df_train["Loan_Amount_Term"].mode()[0],inplace=True)
df_train["Dependents"].fillna(df_train["Dependents"].mode()[0],inplace=True)
df_train["Credit_History"].fillna(df_train["Credit_History"].mode()[0],inplace=True)

#All values of "Dependents" columns were of "str" form now converting to "int" form.
df_train["Dependents"] = df_train["Dependents"].replace('3+',int(3))
df_train["Dependents"] = df_train["Dependents"].replace('1',int(1))
df_train["Dependents"] = df_train["Dependents"].replace('2',int(2))
df_train["Dependents"] = df_train["Dependents"].replace('0',int(0))

df_train["LoanAmount"].fillna(df_train["LoanAmount"].median(),inplace=True)

#df_train["Loan_Status"].fillna(df_train["Loan_Status"].mode()[0],inplace=True)


print(df_train.isnull().sum())
# %%
#Filling all Nan values with mode of respective variable
df_test["Gender"].fillna(df_test["Gender"].mode()[0],inplace=True)
df_test["Married"].fillna(df_test["Married"].mode()[0],inplace=True)
df_test["Self_Employed"].fillna(df_test["Self_Employed"].mode()[0],inplace=True)
df_test["Loan_Amount_Term"].fillna(df_test["Loan_Amount_Term"].mode()[0],inplace=True)
df_test["Dependents"].fillna(df_test["Dependents"].mode()[0],inplace=True)
df_test["Credit_History"].fillna(df_test["Credit_History"].mode()[0],inplace=True)

#All values of "Dependents" columns were of "str" form now converting to "int" form.
df_test["Dependents"] = df_test["Dependents"].replace('3+',int(3))
df_test["Dependents"] = df_test["Dependents"].replace('1',int(1))
df_test["Dependents"] = df_test["Dependents"].replace('2',int(2))
df_test["Dependents"] = df_test["Dependents"].replace('0',int(0))

df_test["LoanAmount"].fillna(df_test["LoanAmount"].median(),inplace=True)

print(df_test.isnull().sum())


# In[319]:


sns.countplot(x = df_train['Gender'], hue = df_train['Married'])
df_train.head()


# In[320]:


####################################################################################################
#Getting log value train :->

df_train["ApplicantIncome"] = np.log(df_train["ApplicantIncome"])
#As "CoapplicantIncome" columns has some "0" values we will get log values except "0"
df_train["CoapplicantIncome"] = [np.log(i) if i!=0 else 0 for i in df_train["CoapplicantIncome"]]
df_train["LoanAmount"] = np.log(df_train["LoanAmount"])
####################################################################################################
####################################################################################################
#Getting log value test :->

#df_test["ApplicantIncome"] = np.log(df_test["ApplicantIncome"])
#As "CoapplicantIncome" columns has some "0" values we will get log values except "0"
#df_test["CoapplicantIncome"] = [np.log(i) if i!=0 else 0 for i in df_test["CoapplicantIncome"]]
#df_test["LoanAmount"] = np.log(df_test["LoanAmount"])
####################################################################################################


# In[321]:


le = preprocessing.LabelEncoder()
df_train['Married'] = le.fit_transform(df_train['Married'])
df_train['Education'] = le.fit_transform(df_train['Education'])
df_train['Self_Employed'] = le.fit_transform(df_train['Self_Employed'])
df_train['Property_Area'] = le.fit_transform(df_train['Property_Area'])
df_train['Gender'] = le.fit_transform(df_train['Gender'])
df_train['Loan_Status'] = le.fit_transform(df_train['Loan_Status'])
# Label encoding test data since all these features have some priority


df_test['Married'] = le.fit_transform(df_test['Married'])
df_test['Education'] = le.fit_transform(df_test['Education'])
df_test['Self_Employed'] = le.fit_transform(df_test['Self_Employed'])
df_test['Property_Area'] = le.fit_transform(df_test['Property_Area'])
df_test['Gender'] = le.fit_transform(df_test['Gender'])


# In[322]:


df_train.head()


# In[323]:


sns.countplot(x = df_train['Gender'], hue = df_train['Loan_Status'])


# In[324]:


fig, ax = plt.subplots(2, 4, figsize = (20, 10))
sns.countplot(x = df_train['Gender'], ax = ax[0][0])
sns.countplot(x = df_train['Married'], ax = ax[0][1])
sns.countplot(x = df_train['Dependents'], ax = ax[0][2])
sns.countplot(x = df_train['Education'], ax = ax[0][3])
sns.countplot(x = df_train['Self_Employed'], ax = ax[1][0])
sns.countplot(x = df_train['Loan_Amount_Term'], ax = ax[1][1]).set_yscale('log')
sns.countplot(x = df_train['Credit_History'], ax = ax[1][2])
sns.countplot(x = df_train['Property_Area'], ax = ax[1][3])


# In[325]:


df_test.head()
####################################################################################################
#Getting log value test :->

#df_test["ApplicantIncome"] = np.log(df_test["ApplicantIncome"])
#As "CoapplicantIncome" columns has some "0" values we will get log values except "0"
#df_test["CoapplicantIncome"] = [np.log(i) if i!=0 else 0 for i in df_test["CoapplicantIncome"]]
#df_test["LoanAmount"] = np.log(df_test["LoanAmount"])
####################################################################################################

df_test.head()


# In[ ]:





# In[326]:


df_train.to_csv('c:\\cit\\CategorizedLoanDataTRAIN.csv', index = False)


# In[327]:


df_test.to_csv('c:\\cit\\CategorizedLoanDataTEST.csv', index = False)


# In[328]:


# Plot number of Approved and Rejected in Train Data
A = list(df_train.Loan_Status).count(1)
B = list(df_train.Loan_Status).count(0)
print("Count of 1<Approved>: ",A,"\nCount of 0<Rejected>: ",B)

fig = px.bar((A,B),x=["Approved","Rejected"],y=[A,B],color=[A,B])
fig.show()


# In[329]:


# no need for loanID

df_train.drop('Loan_ID', axis = 1, inplace = True)
test_ID = df_test['Loan_ID']
df_test.drop('Loan_ID', axis = 1, inplace = True)

# Splitting target variable

y = df_train['Loan_Status']
X = df_train.drop('Loan_Status', axis = 1)


# In[330]:


#SMOTE for upsampling

from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy = 1 ,k_neighbors = 5, random_state=1)
X, y = sm.fit_sample(X, y)

X.shape


# In[331]:


# Normalizing data

normalized_X = preprocessing.normalize(X)
normalized_X_test = preprocessing.normalize(df_test)

models = []


# In[332]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
#use keras to build model
models = keras.Sequential()
models.add(keras.layers.Dense(64, activation=tf.nn.relu))
models.add(keras.layers.Dense(64, activation=tf.nn.relu))
models.add(keras.layers.Dense(1))
models.compile(optimizer = tf.keras.optimizers.RMSprop(0.001), loss='mse', metrics=['mae', 'mse'])

#train model
EPOCHS = 200
history = models.fit(X_train, y_train,epochs = EPOCHS, validation_split = 0.2, verbose=0)
print('Done')
    
#Use model for Prediction of Loan eligibility
ypred = models.predict(normalized_X_test)

# produce an output to inspect Predictions on Test data
result = []
for value in ypred:
    if value == 1:
        result.append('Y')
    else:
        result.append('N')
df = pd.concat([test_ID, pd.DataFrame(result)], axis = 1)
df.rename(columns = {0:'Loan_Status'}, inplace = True)
df.to_csv('c:\\cit\\Final_result.csv', index = False)    
    
#save model as .h5 file for implementation in WebUI
models.save('c:\\cit\\loan_model.h5') 


# In[ ]:





# In[157]:






# In[158]:





# In[ ]:





# In[ ]:




