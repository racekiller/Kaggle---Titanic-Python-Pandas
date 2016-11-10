# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:12:04 2016

@author: JVivas
"""
import csv as csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

Windows_Path = 'C:/Users/jvivas/Dropbox/Private/Personal'\
                '/Github/Kaggle---Titanic-Python-Pandas'
Mac_Path = '/Users/jvivas/Documents/GitHub/Kaggle - Titanic Python Pandas'
Path = Windows_Path
data = []

# For .read_csv, always use header=0 when you know row 0 is the header row
df_original_train = pd.read_csv(Path + '/' + 'train.csv', header=0)
df_original_test = pd.read_csv(Path + '/' + 'test.csv', header=0)
df_test = pd.read_csv(Path + '/' + 'test.csv', header=0)
df_train = pd.read_csv(Path + '/' + 'train.csv', header=0)
df = pd.read_csv(Path + '/' + 'train.csv', header=0)

def  measure_perfomance(x,y,tree,show_accuracy=True,
                        show_classification_report=False,
                        show_confusion_matrix=False):
    y_pred = tree.predict(x)
    if show_accuracy:
        print ("Accuracy:{0:3f}".format(metrics.accuracy_score(y,y_pred)),"/n")
    if show_classification_report:
        print ("Classification_report")
        print (metrics.classification_report(y,y_pred),"/n")
    if show_confusion_matrix:
        print ("Confusion matrix")
        print (metrics.confusion_matrix(y,y_pred), "/n")


# let's see the count of missing data aka. null
df.isnull().sum()
df_test.isnull().sum()

# lets create a new dataframe droping all ages I dont want to come up with
# fake ages (averages according to the class)
df = df.dropna(subset=['Age'])
df_test = df_test.dropna(subset=['Age'])
df_test = df_test.dropna(subset=['Fare'])

# We need to encode the Sex from the dataframe
# Next line will assing an integer to each label  strin from 0 to n
class_mapping = {label: idx for idx,
                 label in enumerate(np.unique(df['Sex']))}

# Next we can use the mapping dictionary to transform the class labels
# into integers
df['Sex'] = df['Sex'].map(class_mapping)
df_test['Sex'] = df_test['Sex'].map(class_mapping)

# We will get one feature for total family members
df['Family'] = df['SibSp'] + df['Parch']
df_test['Family'] = df_test['SibSp'] + df_test['Parch']

# Delete the unnecessary features, that dont add values
df = df.drop(['Name', 'Ticket', 'Cabin',
              'Embarked', 'SibSp', 'Parch'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin',
                        'Embarked', 'SibSp', 'Parch'], axis=1)

# Another way to do One Hot encoder is using get_dummies using pandas
# function
# the function will do the coding automatically and using the dataframe
# not need to convent to numpy array
# This function works only for label string to apply one hot encoder
# to integers we need to do the sklearn one_hot_encoder
#df = pd.get_dummies(df[['PassengerId', 'Survived', 'Pclass',
#                        'Sex', 'Age', 'Fare']])
#df_test = pd.get_dummies(df_test[['PassengerId', 'Pclass',
#                                  'Sex', 'Age', 'Fare']])

# sklearn one hot encoder works only for numpy arrays
dfArray = df.values
dfTestArray = df_test.values

# We use categorical_features to apply the one hot to specific feature
ohe = OneHotEncoder(categorical_features=[2])
dfArray = ohe.fit_transform(dfArray).toarray()
ohe = OneHotEncoder(categorical_features=[1])
dfTestArray = ohe.fit_transform(dfTestArray).toarray()

# Here we assign the survived column to the 'y' and the rest to X
X = np.delete(dfArray, 4, axis=1)
X = np.delete(X, 3, axis=1)
y = dfArray[:, [4]]
X_test_final = np.delete(dfTestArray, 3, axis=1)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, random_state=0)

# Bringing features onto the same scale this is good to do only for non
# tree classifier
# Using normalization
#from sklearn.preprocessing import MinMaxScaler
#mms = MinMaxScaler()
#X_train_norm = mms.fit_transform(X_train)
#X_test_norm = mms.transform(X_test)
#
#from sklearn.preprocessing import StandardScaler
#stdsc = StandardScaler()
#X_train_std = stdsc.fit_transform(X_train)
#X_test_std = stdsc.transform(X_test)

# Train tree decision classifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                              min_samples_leaf=5)
tree = tree.fit(X_train, y_train)
print ('Accuracy for Training Data all features: ',
       tree.score(X_train, y_train))
#tree_norm = tree.fit(X_train_norm, y_train)
#tree_std = tree.fit(X_train_std, y_train)

from sklearn.tree import export_graphviz
from sklearn import metrics
export_graphviz(tree, out_file='tree.dot')
#export_graphviz(tree_norm, out_file='tree_norm.dot')
#export_graphviz(tree_std, out_file='tree_std.dot')

# Measure performance for Tree Classifier
measure_perfomance(X_test, y_test, tree)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10000, random_state=33)
clf = clf.fit(X_train, y_train)
print('Random Forest Accuracy Training Dataset with all features: ',
      clf.score(X_test, y_test))

# Meaningful features using Random Forest
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d %-*s %f" % (f + 1, 30,
         'X_train'+ str([f]), importances[f]))

X_train_new = forest.transform(X_train, threshold=0.10)
# print(X_train_new.shape)
tree = tree.fit(X_train, y_train)
print('Accuracy Training Dataset All features: ',
      tree.score(X_train, y_train))
tree = tree.fit(X_test, y_test)
print('Accuracy Test Dataset with All Features: ',
      tree.score(X_test, y_test))
# lets delete the low performance features from the test dataset
X_test_new = X_test[:, [3, 4, 5]]
tree = tree.fit(X_train_new, y_train)
print('Accuracy Training Dataset with High Performance Features: ',
      tree.score(X_train_new, y_train))
tree = tree.fit(X_test_new, y_test)
print('Accuracy Test Dataset with High Performance Features: ',
      tree.score(X_test_new, y_test))

y_pred_output = tree.predict(X_test_final)
# Collect the test data's PassengerIds before dropping it
ids = df_test['PassengerId'].values
KagglePredictionFile = open("kagglePredictionTitanicTree_V2.csv", "w")
open_file_object = csv.writer(KagglePredictionFile)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(ids, y_pred_output))
KagglePredictionFile.close()
print ('Done.')