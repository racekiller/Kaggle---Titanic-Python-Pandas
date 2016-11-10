# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:34:03 2016

@author: jvivas
"""

import csv as csv
import numpy as np
# Lets do some predictions using Random Forest
# Import the random forest package
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import tree
from sklearn import metrics
import pandas as pd
from sklearn import feature_extraction

Windows_Path = 'C:/Users/jvivas/Dropbox/Private/Personal/Github/Kaggle---Titanic-Python-Pandas'
Mac_Path = '/Users/jvivas/Documents/GitHub/Kaggle - Titanic Python Pandas'
Path = Mac_Path

# For .read_csv, always use header=0 when you know row 0 is the header row
df_original_train = pd.read_csv(Path + '/' + 'train.csv', header=0)
df_original_test = pd.read_csv(Path + '/' + 'test.csv', header=0)
df_test = pd.read_csv(Path + '/' + 'test.csv', header=0)
df = pd.read_csv(Path + '/' + 'train.csv', header=0)

# Lets fill the missing age with the average according to the class where
# they were

# Let see how many missing values we have per feature
df.isnull().sum()

# Lets delete the column we dont need
df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Now lets delete the entire row for the passenger with missing age
# I dont want to come up with a fake age which might lead to unaccurate
# predictions
df = df.dropna()

