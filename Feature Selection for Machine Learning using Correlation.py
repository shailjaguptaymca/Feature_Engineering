#!/usr/bin/env python
# coding: utf-8

# # Feature Selection for Machine Learning using Correlation
# It evaluates subsets of features as: 
# - Good feature subsets contain highly correlated with the target, 
#   .... yet uncorrelated to each other.
#   
# - Using Paribas claims dataset from Kaggle
# 

# ### Brute Force Function
#     - finds correlated features without any further insight.

# ### Finding Groups of Correlated Features
#     - will find groups of correlated features 
#     - and then decide which feature to keep and which to remove

# 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv('train.csv',nrows=50000)
data.shape


# In[3]:


data.head()


# Note: - Feature Selection should be done after pre-processing
#         - After converting categorial to numbers only we can corellate them
#     

# In[6]:


# We will be using only numerical features here
#seelcting variables tht are either integers or float
numerics=['int16','int32','int64','float16','float32','float64']
# selecting values of datatype mentioned in numerics
numerical_vars=list(data.select_dtypes(include=numerics).columns)
data=data[numerical_vars]
data.shape


# In[7]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['target','ID'],axis=1)
y=data['target']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape,x_test.shape


# In[9]:


# visualizing correlated features

corr_mat=x_train.corr()
fig,ax=plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corr_mat)


# ### Brute Force Method to find correlated features
# 
# - To select highly correlated features

# In[ ]:




