# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 18:01:02 2020

@author: Jfitz
"""


#Load libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import tree
#from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.externals.six import StringIO
#from  sklearn.tree import export_graphviz
import pydotplus
import tensorflow as tf
import matplotlib.pyplot as plt

def cleanFile():
    #read the file and remove empty spaces from fields
     dataFile = pd.read_csv("C:\CIT\GROUP\ML\TrainingPredictor.csv")
     #clear file of empty spaces 
     dataFile.columns = dataFile.columns.str.strip()
     #remove columns will null 'nan' values and 'null' values
     dataFile=dataFile.dropna()
     dataFile = dataFile[dataFile.notnull().all(axis = 1)]
     #re classify the fields for numerical values 
     e = dataFile
     #gender_e = pd.Series(LabelEncoder().fit_transform(dataFile["Gender"]))
     #married_e = pd.Series(LabelEncoder().fit_transform(dataFile["Married"]))
    # Dependents_e = pd.Series(LabelEncoder().fit_transform(dataFile["Dependents"]))
     #education_e = pd.Series(LabelEncoder().fit_transform(dataFile["Education"]))
    # Self_Employed_e = pd.Series(LabelEncoder().fit_transform(dataFile["Self_Employed"]))
     #ApplicantIncome_e = pd.Series(LabelEncoder().fit_transform(dataFile["ApplicantIncome"]))
   #  CoapplicantIncome_e = pd.Series(LabelEncoder().fit_transform(dataFile["CoapplicantIncome"]))
   #  LoanAmount_e = pd.Series(LabelEncoder().fit_transform(dataFile["LoanAmount"]))
   #  Loan_Amount_Term_e = pd.Series(LabelEncoder().fit_transform(dataFile["Loan_Amount_Term"]))   
   #  Credit_History_e = pd.Series(LabelEncoder().fit_transform(dataFile["Credit_History"]))
   #  Property_Area_e = pd.Series(LabelEncoder().fit_transform(dataFile["Property_Area"])) 
   #  Loan_Status_e = pd.Series(LabelEncoder().fit_transform(dataFile["Loan_Status"])) 
   #  e=pd.concat([gender_e,married_e,Dependents_e,education_e,Self_Employed_e,ApplicantIncome_e,CoapplicantIncome_e,LoanAmount_e,Loan_Amount_Term_e,Credit_History_e,Property_Area_e, Loan_Status_e], axis=1)     
     e.columns = ['ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','Loan_Amount_Term','LoanAmount','Credit_History','Property_Area','Loan_Status']
     return e


def eligible():

    #call the read file function
    e = cleanFile()
    #take a look at the first 20 lines of data to make sure its there
    #print(e.head(20))

    #In Machine learning we have two kinds of datasets
    #    Training dataset - used to train our model
    #    Testing dataset - used to test if our model is making accurate predictions

    #Our dataset has 480 records. We are going to use 80% of it for training 
    #the model and 20% of the records to evaluate our model.

    # we are only going to use the Income fields, loan amount, loan duration 
    # and credit history fields to train our model.    
    
    #create  dataset from the cleaned file
    loanDataset = e[['ApplicantIncome','CoapplicantIncome','Loan_Amount_Term','LoanAmount','Credit_History','Loan_Status']] 
    
    #test display of datasets
   # print(loanDataset)

    # loanDataset
    #feature = loanDataset[['ApplicantIncome','CoapplicantIncome','Loan_Amount_Term','LoanAmount','Credit_History']] 
    #target = loanDataset[["Loan_Status"]] # target variable for the 'Loan_Status' class attribute
    # loanDataset
    feature = loanDataset[['ApplicantIncome','CoapplicantIncome','Loan_Amount_Term','LoanAmount']] 
    target = loanDataset[["Credit_History"]] # target variable for the 'Credit History'class attribute
    target=target.values
    
    # Split dataset into training set and test set for both datasets
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.7, random_state=3) # 90% training and 10% test
    
    # Create Decision Tree classifer object
    clft = DecisionTreeClassifier()

    clft.fit(X_train, y_train)

    loanScores = model_selection.cross_val_score(clft, X_train, y_train, cv=10)
    #print("***********************************************")
    #print("********** Outputs here ****************")
    #print("***********************************************")    
    # print("\nDataset : ",loanScores)
    #print("DataSet  average: ",loanScores.mean(), " DataSet Standard Deviation: ", loanScores.std())
    
    
    #predictTree=clft.predict(X_test)
    #print("Dataset accuracy score: ",metrics.accuracy_score(predictTree, y_test))
    #nested cross fold validation is best here - validate decision tree
    #parameter_grid = { 'max_depth' : [1,2,3,4,5],
     #                 'max_features': [1,2,3,4],
    #                  'criterion': ["gini", "entropy"]}
   # cross_validation = StratifiedKFold(n_splits=10)

    #find optimised parameters
   # gsclft = GridSearchCV(clft, param_grid = parameter_grid,
    #                           cv = cross_validation)


    #specify cross validation for dataset  
    #cross_validation = StratifiedKFold(n_splits=10)
    #cross_validation.get_n_splits(X_train,y_train)
    #find optimised parameters
    #gsclft = GridSearchCV(clft, param_grid = parameter_grid,
     #                          cv = cross_validation)
    #best estimator is fitted for X_train, y_train (1 and 2)
   # gsclft.fit(X_train, y_train)
    #outer loop with cv 5
    #scoresclft = model_selection.cross_val_score(gsclft, X_train, y_train, cv=5)
    #print(scoresclft.shape)
    #print("Dataset - Nested cross fold accuracy is/Mean Score: ", np.mean(scoresclft))
    #print("Best Score StratifiedKFold: {}".format(gsclft.best_score_))
   # print("Best params: {}".format(gsclft.best_params_))

   # Grid search with 5-fold cross-validation on F1-score
    classifier = RandomForestClassifier(n_estimators = 80)

    
    # re factor y_train to a 1d array to perform prediction
    y = np.ravel(y_train) 
    classifier.fit(X_train, y)
    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    result = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix:")
    #print(result)
    #result1 = classification_report(y_test, y_pred)
   # print("Classification Report:",)
   # print (result1)
    result2 = accuracy_score(y_test,y_pred)
    print("Accuracy:",result2)

    #dataFile = pd.read_csv("C:\CIT\GROUP\ML\TrainingPredictor.csv")
    #clear file of empty spaces 
    #dataFile.columns = dataFile.columns.str.strip()
    #remove columns will null 'nan' values and 'null' values
    #dataFile=dataFile.dropna()
    #dataFile = dataFile[dataFile.notnull().all(axis = 1)]
    #print("***********************************************")
    #print("********** Loan Prediction ****************")
    #print("***********************************************")    
    #series = dataFile["Credit_History"].value_counts()
    #series.plot(kind='bar', figsize=[9,6])
    #plt.xlabel("Credit History in the Training Model recorded")
    #plt.ylabel("No of entries")
    #plt.title("Defaulters v clean histories",fontweight="bold")
    #plt.savefig('task5-MaritalStatus.png')
   # plt.show()
    
cleanFile();
eligible();    