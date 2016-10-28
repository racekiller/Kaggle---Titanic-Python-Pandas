# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:34:03 2016

@author: jvivas
"""

import csv as csv 
import numpy as np

csv_file_object = csv.reader(open('/Users/jvivas/Documents/GitHub/Kaggle - Titanic Python Pandas/train.csv'))
header = csv_file_object.__next__()
data=[]

for row in csv_file_object:
    data.append(row)
data = np.array(data)    

#Look at the first 15 rows of the Age column:   
data[0::,5]

#let's see the dataytype
type(data[0::5,5])

#So, any slice we take from the data is still a Numpy array. Now let's see if we can take the mean of the passenger ages. They will need to be floats instead of strings, so set this up as:

ages_onboard = data[0::,5].astype(np.float)
# we will get an error
# ValueError: could not convert string to float: 
# because there are non numeric values in the column 5
# using numpy array can not do numerical calculations if there are no numerical types in the set
# therefore we need to lcean the data 
# or we cna use pandas which offers more tools to do this kind of tasks (data clenasing)

import pandas as pd
# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('/Users/jvivas/Documents/GitHub/Kaggle - Titanic Python Pandas/train.csv',header=0)
df.head(3)
df.tail(3)

# Showing dataframe type
type(df)
# showing elements type
df.dtypes
# Showing additional information for each elemtn (count and type and if tis null)
df.info()
# Showing statistical information such (mean, max, count, min)
df.describe()

# Data Munging
# One step in any data analysis is the data cleaning. Thankfully pandas makes things easier to filter, manipulate, drop out, fill in, transform and replace values inside the dataframe. Below we also learn the syntax that pandas allows for referring to specific columns.

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
df['Gender'] = 4