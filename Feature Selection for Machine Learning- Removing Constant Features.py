#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#VarianceThreshold function is used for finding constant features 
from sklearn.feature_selection import VarianceThreshold


# In[2]:


data=pd.read_csv('train-santander.csv',nrows=50000)


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


x_train.shape


# In[7]:


x_test.shape


# ### Using variance threshold from sklearn
# Variance threshold is basline approach to feature selection. 
# By default it remove zero variance features i.e.remove all the features having same value in all the samples or rows
# It removes features having threshold varince. Threshold variance is defined by user

# In[8]:


# using variance threshold function on x_train to detect and remove constat features
sel=VarianceThreshold(threshold=0)
sel.fit(x_train)


# In[9]:


#sel.get_support() will return true if no constant feature has been found
#sum(sel.get_support()) will sum total number of such columns
sum(sel.get_support())
#another method:
#len(x_train.columns[sel.get_support()])


# In[10]:


x_test.shape


# In[11]:


#make list of constant features
print(len([
    #for all columns
        x for x in x_train.columns
    #if column value if false i.e constant feature
        if x not in x_train.columns[sel.get_support()]
    ]))


# In[12]:


#this will print all the column names having duplicate features
[x for x in x_train.columns if x not in x_train.columns[sel.get_support()]]


# In[13]:


# to check if the column has unique values
x_train['ind_var2_0'].unique()
#[0]implies no unique value


# In[14]:


# we use transform function to reduce the training and testing sets
x_train=sel.transform(x_train)
x_test=sel.transform(x_test)
x_train.shape, x_test.shape


# # Coding Without Using Variance Threshold function
# 

# In[15]:


data=pd.read_csv('train-santander.csv',nrows=50000)


# In[16]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['TARGET'],axis=1)
y=data['TARGET']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[17]:


#this list capture all the constant
#if std var of feature is constnt
const_features=[x for x in x_train.columns if x_train[x].std()==0]


# In[18]:


len(const_features)


# In[20]:


#axis=1 means to look features among columns
#inplace=true means do modifications in the actual data
x_train.drop(labels=const_features,axis=1,inplace=True)
x_test.drop(labels=const_features,axis=1,inplace=True)


# In[22]:


x_train.shape, x_test.shape


# ### Removing constant features in categorial variables
# One alternative is to encode the categories as numbers and then use the code above. But then you will out effort in pre processing variables that are not informative
# 

# In[23]:


data=pd.read_csv('train-santander.csv',nrows=50000)


# In[24]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['TARGET'],axis=1)
y=data['TARGET']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[25]:


x_train.shape, x_test.shape


# In[28]:


# as the dataset is numerical we convert this datatype to object to show that it is categorial
#this step is not required if ou have categorial data
# it is capital o and not zero as argument
x_train=x_train.astype('O')
x_train.dtypes


# In[29]:


#to find constant features in categorial variable
# i.e. finding those columns that contain only one label

constant_features=[ x for x in x_train.columns if len(x_train[x].unique())==1]
len(constant_features)


# In[ ]:




