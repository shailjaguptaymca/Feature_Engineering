#!/usr/bin/env python
# coding: utf-8

# # Feature Selection- Removing Duplicate Values
# Duplicated values: when one or more features have same values throughout all the rows

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#VarianceThreshold function is used for finding constant features 
from sklearn.feature_selection import VarianceThreshold


# In[2]:


data=pd.read_csv('train-santander.csv',nrows=15000)


# In[3]:


data.shape


# In[4]:


# to check if there are no missing values in entire dataset
[col for col in data.columns if data[col].isnull().sum()>0]


# In[5]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['TARGET'],axis=1)
y=data['TARGET']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[6]:


x_train.shape, x_test.shape


# ### Finding duplicated values in small dataset
# 
# We use duplicated function to identify duplicate rows. We use this func, by transposing the dataframe such that rows are now columns.

# In[7]:


# create a transposed dataframe
data_t=x_train.T
data_t.head()


# In[8]:


# find all the duplicated columns
data_t.duplicated().sum()


# In[10]:


#visualize the duplicated rows
data_t[data_t.duplicated()]


# In[12]:


# we can catpure duplicated features by capturing the index values 
#... of transosed dataset
duplicated_features=data_t[data_t.duplicated()].index.values
duplicated_features


# In[13]:


#drop duplicated features and transpose dataset to orignal shape
data_unique=data_t.drop_duplicates(keep='first').T
data_unique.shape


# In[14]:


# you can find columns not present in orignal dataset to find duplictaed columns
dup_feat=[col for col in data.columns if col not in data_unique.columns]
dup_feat


# ### Finding duplicated values in large dataset

# In[24]:


data=pd.read_csv('train-santander.csv',nrows=50000)


# In[25]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['TARGET'],axis=1)
y=data['TARGET']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[26]:


x_train.shape, x_test.shape


# In[23]:


duplicated_feat=[]
for i in range (0,len(x_train.columns)):
    if i%10==0:
        print(i)#this is done so see how loop works
    col_1=x_train.columns[i]
    for col_2 in x_train.columns[i+1:]:
        if x_train[col_1].equals(x_train[col_2]):
            duplicated_feat.append(col_2)


# In[27]:


#print total number of duplicated columns
print(len(set(duplicated_feat)))


# In[28]:


#printing list of duplicated features
set(duplicated_feat)


# In[30]:


# we are comparing two columns to show set od duplicated columns
duplicated_feat=[]
for i in range(0,len(x_train.columns)):
    col_1=x_train.columns[i]
    for col_2 in x_train.columns[i+1:]:
        if x_train[col_1].equals(x_train[col_2]):
            print(col_1)
            print(col_2)
            print()
            
            duplicated_feat.append(col_2)


# In[31]:


# to see if the pairs are actually showing equal values
# to check that those features are duplicated
# select a random pair

x_train[['ind_var2_0','ind_var28_0']].head(10)


# In[ ]:




