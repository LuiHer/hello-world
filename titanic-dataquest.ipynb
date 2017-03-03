# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:15:45 2017

@author: hernanl2
"""

#install seaborn in from spyder ipython console using !pip install seaborn
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#source: https://www.dataquest.io/mission/74/getting-started-with-kaggle/9/making-predictions-with-scikit-learn
#import the linear regression class
from sklearn.linear_model import LinearRegression

#SKlearn also has a helper for cross-validation
from sklearn.cross_validation import KFold

titanic = pd.read_csv('../input/train.csv')

'''
#Step 1: Examine the Data
print (titanic.head(5))
print (titanic.describe())

#useful code for finding values and counts
#print(titanic['Embarked'].unique())

#count unique items
#from collections import Counter
#c = Counter(titanic['Embarked'])
#print (c.items())

#second option, will not return nan counts
#print titanic['Embarked'].value_counts()

'''
#Step 2 Handle missing data: A.Fill missing age data with median values
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

#Step 2B: Fill empty Embarked values with mode value
titanic['Embarked'] = titanic['Embarked'].fillna('S')

#Step 3 Handling Non-numeric values:
#A. ignore Ticket, Cabin, and Name columns
#B. convert Sex to numeric
#print(titanic['Sex'].unique())
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

#C. Change embarked locations to numeric values
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2


#Step 4: A. train algorithm LINEAR REGRESSION (different data set than test to avoid overfitting)
#Step 4B ;Cross-validating split into parts ('folds'), say 3, and train on 1,2 predict on3; train on 1 and 3, predict on second, train on 2 and 3 and predict on first. 
#see imports at top

#The columns we'll use to predict the target
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

#initialize our algorithm class
alg= LinearRegression()
#Generate cross-validation folds for the titanic data set
#It returns the row indices corresponding to train and test
#We set random_state to ensure we get the same splits every time we run this
kf = KFold(titanic.shape[0], n_folds=3, random_state =1)

predictions = []

for train, test in kf:
    #The predictos we're using to train the algorith,
    #Note how we only take the rows in the train folds
    train_predictors = (titanic[predictors].iloc[train,:])
    #The target we're using to train the algorithm 
    train_target = titanic['Survived'].iloc[train]
    #Training the algorithm using the predictors and target
    alg.fit(train_predictors, train_target)
    #We can now make test predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

#Step 5: Evaluating prediction error
#A. Error metric (described by Kaggle), %correct predictions
#the predictions are in three separate NumPy arrays; concatenate into single array
#we concatenate on axis 0 because htey only have on eaxis
predictions = np.concatenate(predictions, axis = 0)

#Map predictions to outcomes (the only possible outcome are 1 and 0)
predictions[predictions >.5] = 1
predictions[predictions <=.5] = 0

#proportion predictions match 'Survived' data
#Add Predictoins columns to dataframe
titanic['Predicted'] = predictions

#number of rows where predictions match Survival 
n_match = float(len(titanic[['Survived','Predicted']][(titanic['Survived'] == titanic['Predicted'])]))

#convert to percentage
accuracy = (n_match/len(titanic))

###answer code
#accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
