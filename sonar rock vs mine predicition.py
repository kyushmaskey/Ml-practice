#!/usr/bin/env python
# coding: utf-8

# Load the libraries
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Loading the file from csv file  if file have no header use header=None
# 

# In[10]:


sonar_data = pd.read_csv(r'C:\Users\kyush\Desktop\Copy of sonar data.csv', header= None)


# In[11]:


sonar_data.head()


# In[12]:


sonar_data.shape


# Mean median statistical calculation of data

# In[13]:


sonar_data.describe() 


# In[15]:


sonar_data[60].value_counts() # it calculates the number of R and M sonar_data[60] 60 means the indes value

M--> mine
R--> rock


# In[16]:


sonar_data.groupby(60).mean()


# separating data and labels

# In[19]:


X = sonar_data.drop(columns=60, axis =1 )
Y = sonar_data[60]


# In[20]:


print(X)


# In[21]:


print(X,Y)


# In[22]:


print(X)
print(Y)


# Training and test data

# In[23]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.1,stratify=Y, random_state=1)


# In[24]:


print(X.shape,X_train.shape, X_test.shape)


# In[25]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state=1)


# In[26]:


print(X.shape,X_train.shape, X_test.shape)


# In[27]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2,stratify=Y, random_state=2)
print(X.shape,X_train.shape, X_test.shape)


# In[31]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.1,stratify=Y, random_state=1)
print(X.shape,X_train.shape, X_test.shape) 
# using this


# In[32]:


print(X_train)
print(Y_train)


# Training the model using Logistic Regression

# In[29]:


model= LogisticRegression()


# In[34]:


model.fit(X_train, Y_train)


# Model Evaluation

# In[35]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[36]:


print (training_data_accuracy)


# In[37]:


X_test_prediction = model.predict(X_test)
training_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[38]:


print ('Accuracy on test data :' ,training_data_accuracy)


# Making a Predicitive System

# In[43]:


input_data=[0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032]

# changing the input_data to numpy array

input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one instance
input_data_reshaped =  input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print('The object is Rock')
else:
    print('The object is Mine')


# In[ ]:




