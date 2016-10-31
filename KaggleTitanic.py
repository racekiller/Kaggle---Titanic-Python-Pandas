# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:34:03 2016

@author: jvivas
"""

import csv as csv 
import numpy as np

Windows_Path = 'C:/Users/jvivas/Dropbox/Private/Personal/Github/Kaggle---Titanic-Python-Pandas'
Mac_Path = '/Users/jvivas/Documents/GitHub/Kaggle - Titanic Python Pandas'
Path = Mac_Path
csv_file_object = csv.reader(open(Path+'/' + 'train.csv'))
header = csv_file_object.__next__()
data=[]

for row in csv_file_object:
    data.append(row)
data = np.array(data)    

#Look at the first 15 rows of the Age column:   
data[0::,5]

#let's see the dataytype
type(data[0::5,5])

# So, any slice we take from the data is still a Numpy array. Now let's see if
# we can take the mean of the passenger ages. They will need to be floats
# instead of strings, so set this up as:

# ages_onboard = data[0::,5].astype(np.float)

# we will get an error
# ValueError: could not convert string to float: 
# because there are non numeric values in the column 5
# using numpy array can not do numerical calculations if there are no
# numerical types in the set
# therefore we need to lcean the data 
# or we cna use pandas which offers more tools to do this kind of tasks
# (data clenasing)

import pandas as pd
# For .read_csv, always use header=0 when you know row 0 is the header row
df_original_train = pd.read_csv(Path + '/' + 'train.csv', header=0)
df_original_test = pd.read_csv(Path + '/' + 'test.csv', header=0)
df_test = pd.read_csv(Path + '/' + 'test.csv', header=0)
df = pd.read_csv(Path + '/' + 'train.csv', header=0)
df.head(3)
df.tail(3)

# Showing dataframe type
type(df)
# showing elements type
df.dtypes
# Showing additional information for each elemtn (count and type and if tis
# null)
df.info()
# Showing statistical information such (mean, max, count, min)
df.describe()

# Data Munging
# One step in any data analysis is the data cleaning. Thankfully pandas makes
# things easier to filter, manipulate, drop out, fill in, transform and
# replace values inside the dataframe. Below we also learn the syntax that
# pandas allows for referring to specific columns.

# Referencing and filtering
# Let's acquire the first 10 rows of the Age column. In pandas this is

df['Age'][0:10]
df.Age[0:10]

# let's do some calculations
df['Age'].mean()
df['Age'].median()

# How to show specific columns from the df
df[['Sex','Pclass','Age']]

# How to filter data 
# Show all rows where age is greater than 60
df[df['Age'] > 60]

# Show specific columns that matches the WHERE clause
df[df['Age'] > 60][['Pclass','Age','Survived']]

# Lets take a look to the null value in Ages

df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

# here we will go over the dataframe to get the count of male per class
for i in range(1,4):
    a = len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])
    print (a)
    
# let's draw some picture
import pylab as P
df['Age'].hist()
P.show()

df['Age'].dropna().hist(bins=16, range=(0,80),alpha = 0.5)
P.show()

# Cleaning the data
# Creating a column into df dataframe
df['Gender'] = 4

# Here we take the first letter of the element and convert to Uppercase
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())

# Now here we replace each element string with integer number in this case
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)

# We will do the same than above but this time we will skip the nan (null)
df['Embarked_F'] = df['Embarked'].dropna().map({'S': 0, 'Q': 1, 'C': 2}).\
                    astype(int)
df_test['Embarked_F'] = df_test['Embarked'].dropna().map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
# working with missing values, what to do?
# There are some features that missing values can be converted to null but
# there are others that can not for example the Age of the passenger in this
# case what we can do is take the mean by class

median_ages = np.zeros((2, 3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()

# lets create a new column for Age so we can put in there the new values for
# missing data
df['AgeFill'] = df['Age']
# Lets show the total of passenger with missing Age for specific columns
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

# Lets populate the dataframe with new ages
# here we go over the dataframe and will assign the meadian age that matches
# the Gender and Pclass in order by i and j
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),
               'AgeFill'] = median_ages[i, j]

# We repeat the process for test dataset
median_ages = np.zeros((2, 3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df_test[(df_test['Gender'] == i) & \
                              (df_test['Pclass'] == j+1)]['Age'].dropna().median()

# lets create a new column for Age so we can put in there the new values for
# missing data
df_test['AgeFill'] = df_test['Age']
# Lets show the total of passenger with missing Age for specific columns
df_test[df_test['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

# Lets populate the dataframe with new ages
# here we go over the dataframe and will assign the meadian age that matches
# the Gender and Pclass in order by i and j
for i in range(0, 2):
    for j in range(0, 3):
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df.Pclass == j+1),
               'AgeFill'] = median_ages[i, j]

# Let's also create a feature that records whether the Age was originally
# missing.
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df_test['AgeIsNull'] = pd.isnull(df_test.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

# Deletig unnecesary columns
# Here we can show the columns that matches specific criteria as well
df.dtypes[df.dtypes.map(lambda x: x=='object')]
df_test.dtypes[df_test.dtypes.map(lambda x: x=='object')]
# We can delete these columns 
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
df_test = df_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 
# We can delete the Age column as well since we have created the better AgeFill
df = df.drop(['Age'],axis = 1)
df_test = df_test.drop(['Age'],axis = 1)

#delete null from Embarked  column
df = df[pd.notnull(df['Embarked_F'])]

#delete null from Fare  column (we kept df_test with one null value for a fare record)
# We need to get the average frpice per class and assign it to the null record  
df_test2 = df_test[pd.notnull(df_test['Fare'])]

if len(df_test.Fare[df_test.Fare.isnull()]) > 0:
    fare_Average = np.zeros(3)
    for i in range(0,3):
        fare_Average[i] = df_test[df_test.Pclass == i+1]['Fare'].dropna().median()
    for i in range(0,3):
        df_test.loc[(df_test.Fare.isnull()) & (df_test.Pclass == i+1),\
        ['Fare']] = fare_Average[i]

# Collect the test data's PassengerIds before dropping it
ids = df_test['PassengerId'].values
# The final step is to convert it into a Numpy array. Pandas can always
# send back an array using the .values method. Assign to a new variable
# train_data:

x_train_data = df.ix[:, df.columns != 'Survived'].values
y_train_data = df['Survived'].values
x_test_data = df_test.values

# Lets do some predictions using Random Forest
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree

x_Train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size = 0.25, random_state = 33)

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 100, min_samples_leaf = 5)

clf = clf.fit(x_Train, y_train)

clf_RF = RandomForestClassifier(n_estimators = 10, random_state = 33)
clf_RF = clf_RF.fit(x_Train, y_train)
y_pred_RF = clf_RF.predict(x_test)

from sklearn import metrics
def  measure_perfomance(x,y,clf,show_accuracy=True,show_classification_report=True\
                    ,show_confusion_matrix=True):
    y_pred = clf.predict(x)
    if show_accuracy:
        print ("Accuracy:{0:3f}".format(metrics.accuracy_score(y,y_pred)),"/n")
    if show_classification_report:
        print ("Classification_report")
        print (metrics.classification_report(y,y_pred),"/n")
    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y,y_pred), "/n")

# Measure performance for Tree Classifier
measure_perfomance(x_test,y_test, clf,show_classification_report=True, \
                    show_confusion_matrix=True)

# Measure performance for Randomr FOrest Classifier
measure_perfomance(x_test,y_test, clf_RF,show_classification_report=True, \
                    show_confusion_matrix=True)

y_pred_RF_output = clf_RF.predict(x_test_data).astype(int)

# Sending data to CSV file using Kaggle code
KagglePredictionFile = open("kagglePredictionTitanic.csv", "wb")
open_file_object = csv.writer(KagglePredictionFile)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, y_pred_RF_output))
KagglePredictionFile.close()
print ('Done.')

