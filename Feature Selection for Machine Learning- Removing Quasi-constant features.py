#!/usr/bin/env python
# coding: utf-8

# # Quasi-constant features
# -features that show same value for a great majority of observations of thedataset. 
# 
# -it is generally first step in feature selection
# 
# -we will use customer satisfaction dataset
# 
# - identify quasi-constant features using variance threshold and manually

# In[2]:


#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


# In[6]:


#load train_santander dataset
# loading 50000 rows
data=pd.read_csv("train-santander.csv", nrows=50000)
data.shape


# In[8]:


# Note: we should do feature engineering before feature selection
 
# i.e check the presence of null data 
# i.e is any column present that has null values
[col for col in data.columns if data[col].isnull().sum()>0]


# -Note: feature selection procedure, it is good practice to select the features by examining only the training set.
# 
# -This is done to avoid overfiting

# In[11]:


#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['TARGET'],axis=1)
y=data['TARGET']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape, x_test.shape


# Remove constant features before removing quasi-constant features

# In[12]:


#this list capture all the constant
#if std var of feature is constnt
const_features=[x for x in x_train.columns if x_train[x].std()==0]


# In[13]:


#axis=1 means to look features among columns
#inplace=true means do modifications in the actual data
x_train.drop(labels=const_features,axis=1,inplace=True)
x_test.drop(labels=const_features,axis=1,inplace=True)


# In[14]:


x_train.shape, x_test.shape


# ## Removing Quasi-constant features

# ### Using variance threshold function

# In[17]:


# we will remove quasi constant by setting some threshold other than zero
# 0.1 indicate 99% of observations approximately
# i.e. feture that show same value for 99% of observtions
sel=VarianceThreshold(threshold=0.01)
#fit finds the features with low variance i.e quasi constant ones
#fitting variance with training dataset
sel.fit(x_train)


# In[18]:


# get_support is a boolean that return true if feature is not quasi-constant
# if we sum over get_support, we get the number of features that are not quasi-constant
# no. of fetures that has to be retained 
sum(sel.get_support())


# In[20]:


# to print the total number of quasi-constant features acc. to threshold
print(
        len([
            x for x in x_train.columns
            if x not in x_train.columns[sel.get_support()]
        ]))


# In[22]:


#print quasi constant features
[x for x in x_train.columns if x not in x_train.columns[sel.get_support()]]


# In[24]:


# percentage of observations showing each of the different values
x_train['ind_var31'].value_counts()/np.float(len(x_train))
# this will shows the number of each vaue present in the column
# we will see that >99% of observations show one value 0
# Therefore, this features is almost constant


# In[25]:


# removing features having quasi-constant features
x_train=sel.transform(x_train)
x_test=sel.transform(x_test)
x_train.shape, x_test.shape


# In[26]:


# by removing constant and quasi-constant we have removed over 100 features from present dataset


# # Coding manually

# In[28]:


data=pd.read_csv("train-santander.csv", nrows=50000)
data.shape

#selecting x by dropping target column and y by selecting target column
x=data.drop(labels=['TARGET'],axis=1)
y=data['TARGET']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape, x_test.shape


# In[29]:


#this list capture all the constant
#if std var of feature is constnt
const_features=[x for x in x_train.columns if x_train[x].std()==0]


# In[30]:


#axis=1 means to look features among columns
#inplace=true means do modifications in the actual data
x_train.drop(labels=const_features,axis=1,inplace=True)
x_test.drop(labels=const_features,axis=1,inplace=True)


# In[31]:


x_train.shape, x_test.shape


# In[32]:


# how to take quasi-constant features
quasi_constant_feat=[]
for feature in x_train.columns:
    #find predominant value
    #by selcting features by counting all and dividing by total number of features
    #sort values from bigger values to smaller
    # and values[0], value that show highest number of observation
    predominant=(x_train[feature].value_counts()/np.float(
    len(x_train))).sort_values(ascending=False).values[0]
    
    if predominant > 0.998:
        quasi_constant_feat.append(feature)
        
len(quasi_constant_feat)


# In[33]:


# select the first one from the list
quasi_constant_feat[0]


# In[34]:


x_train['imp_op_var40_efect_ult1'].value_counts()/np.float(len(x_train))


# In[ ]:




