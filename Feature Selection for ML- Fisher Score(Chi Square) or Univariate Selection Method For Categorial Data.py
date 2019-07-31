#!/usr/bin/env python
# coding: utf-8

# In[1]:


# using titanic dataset


# In[ ]:


# Fisher Score: The smaller the p value, the more significant the feature
#.. is to predict the target, in this case Survival in the titanic


# In[2]:


# this score is to evaluate categorial variables in classification task
# for binary target
#


# In[3]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile


# In[4]:


data=pd.read_csv('train.csv',nrows=50000)
data.shape


# In[5]:


data.head()


# In[6]:


# we will be working on target 'Sex'
# we will encode the labels of the catefories into numbers

# for sex/gender
data['Sex']=np.where(data.Sex == 'male',1,0)

#for Embarked
ordinal_label={k: i for i,k in enumerate(data['Embarked'].unique(),0)}
data['Embarked']=data['Embarked'].map(ordinal_label)

# Pclass is already ordinal


# In[9]:


#selecting x by dropping target column and y by selecting target column
x=data[['Pclass','Sex','Embarked']]
y=data['Survived']
#selecting train and test
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape, x_test.shape


# In[10]:


# calculate fisher score for each feature with the target
# it will retuen two array: F-Score and p value
# the F-score will then be evaluated against the chi2 distribution to obtain p value

f_score=chi2(x_train.fillna(0),y_train)
f_score


# In[ ]:


# the p value is the indication of how different are the distribution
#..of the classes of the target among the different labels of the features
# smaller p value the more different the distribution


# In[12]:


# adding variable name for clearer visualization
pvalues= pd.Series(f_score[1])
# index are the columns
pvalues.index=x_train.columns
# sorting from highest to lowest p values
pvalues.sort_values(ascending=False)


# In[ ]:


# we see sex is important feature than pclass than embarked
# you can combine this procedure SelectKBest or SelectPercent as in mutual
#.. information 


# In[ ]:


# in a large dataset, alot of features will show smaller p value.
# ...This do not indicate best features. Take care while selecting features.
#.. The ultra small value of p will show that there are too many features present

