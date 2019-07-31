#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
 
from sklearn.metrics import roc_auc_score


# In[2]:


# load the Santander customer satisfaction dataset from Kaggle
 
data = pd.read_csv('train-santander.csv')
data.shape


# In[3]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['TARGET'], axis=1),
    data['TARGET'],
    test_size=0.3,
    random_state=0)
 
X_train.shape, X_test.shape


# In[4]:


# I keep a copy of the dataset with all the variables
# to measure the performance of machine learning models
# at the end of the notebook
 


# In[5]:


X_train_original = X_train.copy()
X_test_original = X_test.copy()


# # remove constant features

# In[6]:


constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]
 
X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)
 
X_train.shape, X_test.shape


# # remove quasi-constant features

# In[7]:


sel = VarianceThreshold(
    threshold=0.01)  # 0.1 indicates 99% of observations approximately
 
sel.fit(X_train)  # fit finds the features with low variance
 
sum(sel.get_support()) # how many not quasi-constant?


# In[8]:


features_to_keep = X_train.columns[sel.get_support()]


# In[9]:


X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
 
X_train.shape, X_test.shape


# In[10]:


# sklearn transformations lead to numpy arrays
# here I transform the arrays back to dataframes
# please be mindful of getting the columns assigned
# correctly


# In[11]:


X_train= pd.DataFrame(X_train)
X_train.columns = features_to_keep
 
X_test= pd.DataFrame(X_test)
X_test.columns = features_to_keep


# # check for duplicated features in the training set

# In[12]:


duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # this helps me understand how the loop is going
        print(i)
 
    col_1 = X_train.columns[i]
 
    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)
            
len(duplicated_feat)


# In[13]:


# remove duplicated features
X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
X_test.drop(labels=duplicated_feat, axis=1, inplace=True)
 
X_train.shape, X_test.shape


# In[14]:


# I keep a copy of the dataset except constant and duplicated variables
# to measure the performance of machine learning models
# at the end of the notebook
X_train_basic_filter = X_train.copy()
X_test_basic_filter = X_test.copy()


# # find and remove correlated features

# In[21]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
 
corr_features = correlation(X_train, 0.8)
print('correlated features: ', len(set(corr_features)) )


# In[22]:


X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)
 
X_train.shape, X_test.shape


# In[23]:


# Compare performance in machine learning models


# In[24]:


# create a function to build random forests and compare performance in train and test set
 
def run_randomForests(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
    rf.fit(X_train, y_train)
    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[25]:


# original
run_randomForests(X_train_original.drop(labels=['ID'], axis=1),
                  X_test_original.drop(labels=['ID'], axis=1),
                  y_train, y_test)


# In[26]:


# filter methods - basic
run_randomForests(X_train_basic_filter.drop(labels=['ID'], axis=1),
                  X_test_basic_filter.drop(labels=['ID'], axis=1),
                  y_train, y_test)


# In[27]:


# filter methods - correlation
run_randomForests(X_train.drop(labels=['ID'], axis=1),
                  X_test.drop(labels=['ID'], axis=1),
                  y_train, y_test)


# In[ ]:


#We can see that removing constant, quasi-constant, duplicated and correlated features
#...reduced the feature space dramatically (from 371 to 119),
#...without affecting the performance of the random forests (0.790 vs 0.794). 
#..If anything else, the model can now make even better predictions. 
#...And this is most likely due to the fact that high feature spaces affect negatively the performance of random forests.


# In[30]:



# create a function to build logistic regression and compare performance in train and test set
 
def run_logistic(X_train, X_test, y_train, y_test):
    # function to train and test the performance of logistic regression
    logit = LogisticRegression(random_state=44)
    logit.fit(X_train, y_train)
    print('Train set')
    pred = logit.predict_proba(X_train)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = logit.predict_proba(X_test)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# In[31]:


# original
# for logistic regression features need to be in the same scale
 
# original
scaler = StandardScaler().fit(X_train_original.drop(labels=['ID'], axis=1))
 
run_logistic(scaler.transform(X_train_original.drop(labels=['ID'], axis=1)),
             scaler.transform(X_test_original.drop(labels=['ID'], axis=1)), y_train, y_test)


# In[32]:


# filter methods - basic
scaler = StandardScaler().fit(X_train_basic_filter.drop(labels=['ID'], axis=1))
 
run_logistic(scaler.transform(X_train_basic_filter.drop(labels=['ID'], axis=1)),
             scaler.transform(X_test_basic_filter.drop(labels=['ID'], axis=1)),
                  y_train, y_test)


# In[33]:


# filter methods - correlation
scaler = StandardScaler().fit(X_train.drop(labels=['ID'], axis=1))
 
run_logistic(scaler.transform(X_train.drop(labels=['ID'], axis=1)),
             scaler.transform(X_test.drop(labels=['ID'], axis=1)),
                  y_train, y_test)


# In[ ]:


### Similarly, for logistic regression, removing constant, quasi-constant, duplicated and highly correlated features, did not
###...affect dramatically the performance of the algorithm.


# In[ ]:




