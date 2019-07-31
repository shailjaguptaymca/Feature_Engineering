#!/usr/bin/env python
# coding: utf-8

# ### Classification 
# - It will use Paribas Claim dataset

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile


# In[2]:


data=pd.read_csv('paribas.csv',nrows=50000)
data.shape


# In[3]:


data.head()


# In[4]:


# feature selection should be done after all values are converted to numbers
# here we are selecting only the numerical columns 


# In[6]:


numerics=['int16','int32','int64','float16','float32','float64']
# select all the columns having datatypes mentioned in numerics
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape


# In[ ]:





# In[8]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['target','ID'],axis=1)
y=data['target']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape, x_test.shape


# In[ ]:


# calculate mutual information between the variables and the target
# this will return mutual information of each feature
# smaller the value the less information the feature has about the target


# In[10]:


# passing train and target set
# fillna(0) is filling of empty cells with zero
mi= mutual_info_classif(x_train.fillna(0), y_train)


# In[12]:


#mi of each feature and the target
mi


# In[13]:


# creating series with these values
mi= pd.Series(mi)
# where the index of the series is name of the columns
mi.index=x_train.columns
# sorting the series in descending order i.e highest to lowest MI
mi.sort_values(ascending=False)


# In[15]:


#creating bar plot of these values
mi.sort_values(ascending=False).plot.bar(figsize=(20,8))


# In[ ]:


# we see that there are feature contributing a lot to the target(on left side)


# In[ ]:


# we can set a point(till where we want features) like..
# top 10 features, top 20 features or top 10percentile features...


# In[16]:


# for this we can use mutual information with SelectKBest or SelectPercentile
# SelectKBest- allows you to determine how many features to select
# SelectPercentile- allows you the features within a certain percentile.


# In[18]:


# to select top 10 features
# scoring method used: nutual_info_classif
# k are number of features
# get_support method is true/false indicatio tht tell which feature to 
#.... extract is True, False is for features to eliminate
sel=SelectKBest(mutual_info_classif, k=10).fit(x_train.fillna(0), y_train)
x_train.columns[sel_.get_support()]


# ### Regression
# 
# - House Price dataset

# In[21]:


data=pd.read_csv('train.csv',nrows=50000)
data.shape


# In[22]:


numerics=['int16','int32','int64','float16','float32','float64']
# select all the columns having datatypes mentioned in numerics
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape


# In[23]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['SalePrice'],axis=1)
y=data['SalePrice']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape, x_test.shape


# In[24]:


# Calling Mutual Regression for Regression


# In[25]:


# passing train and target set
# fillna(0) is filling of empty cells with zero
mi= mutual_info_regression(x_train.fillna(0), y_train)
# creating series with these values
mi= pd.Series(mi)
# where the index of the series is name of the columns
mi.index=x_train.columns
# sorting the series in descending order i.e highest to lowest MI
mi.sort_values(ascending=False)


# In[28]:


# here i will select the top 10 percentile
# to select features in 10 percentile of total features
# scoring method used: nutual_info_classif
# get_support method is true/false indicatio tht tell which feature to 
#.... extract is True, False is for features to eliminate
sel=SelectPercentile(mutual_info_classif, percentile=10).fit(x_train.fillna(0), y_train)
x_train.columns[sel.get_support()]


# In[ ]:


# we can use select percentile.transform to reduce the feature space


# In[ ]:




