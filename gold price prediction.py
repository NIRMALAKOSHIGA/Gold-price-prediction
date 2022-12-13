#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Data collection and processing

# In[2]:


#loading the csv data to a pandas dataframe
data=pd.read_csv('gld_price_data.csv')


# In[3]:


#print frist 5 rows in the dataframe
data.head()


# In[4]:


#print last 5 rows in the dataframe
data.tail()


# In[5]:


#number of row and columns
data.shape


# In[6]:


#getting some basic information about the data
data.info()


# In[7]:


#checking number of missing values
data.isnull().sum()


# In[8]:


#getting the statistical measures of the data
data.describe()


# collelation
# 1.Positive Correlation
# 2.Negative Correlation

# In[9]:


correlation=data.corr()


# In[10]:


#constructing a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Reds')


# In[11]:


#correlation values of GLD
print(correlation ['GLD'])


# In[12]:


#checking the distribution of the GLD price
sns.distplot(data['GLD'],color='red')


# splitting the futures and target

# In[13]:


X= data.drop(['Date','GLD'],axis=1)
Y=data['GLD']


# In[14]:


print(X)


# In[15]:


print(Y)


# Splitting into traing data and test data

# In[16]:


X_train ,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=2)


# Model Traing : Random Forest Regressor

# In[17]:


regressor= RandomForestRegressor(n_estimators=100)


# In[18]:


#traing the model
regressor.fit(X_train,Y_train)


# Model Evalution

# In[19]:


#prediction on test data
test_data_prediction=regressor.predict(X_test)


# In[20]:


print(test_data_prediction)


# In[21]:


#R squared error
error_score=metrics.r2_score(Y_test,test_data_prediction)
print("R squared error:",error_score)


# Compare the Actual Values and Predicted Values in Plot

# In[22]:


Y_test=list(Y_test)


# In[23]:


plt.plot(Y_test,color='blue',label='Actual Value')
plt.plot(test_data_prediction,color='green',label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[ ]:




