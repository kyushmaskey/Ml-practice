#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[4]:


#loading the dataset
diabetes_dataset = pd.read_csv( 'C:/Users/kyush/Desktop/diabetes.csv')


# In[5]:


diabetes_dataset.head()
# 1 --> person is diabetic
#  0--> person is non-diabetic


# In[6]:


# getting the number of rows and columns

diabetes_dataset.shape


# In[7]:


# getting the statistical message of the data

diabetes_dataset.describe()


# In[8]:


#checking O and 1 label in Outcome

diabetes_dataset['Outcome'].value_counts()


# In[9]:


# calculating mean value for 0 and 1 
diabetes_dataset.groupby('Outcome').mean()


# In[10]:


# separating the data and labels    X --> data  y--> lables for  dropping the columns use axis = 0

X = diabetes_dataset.drop(columns = 'Outcome' , axis = 1)
Y = diabetes_dataset['Outcome']


# In[12]:


print(X)


# In[13]:


print(Y)


# Data Standardization

# In[14]:


scaler = StandardScaler()


# In[15]:


scaler.fit(X)


# In[16]:


standardized_data = scaler.transform(X)


# In[17]:


print(standardized_data)


# In[19]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[20]:


print(X,Y)


# In[21]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape,X_train.shape,X_test.shape)


# Training the Model

# In[24]:


classifier = svm.SVC(kernel= 'linear')


# In[26]:


#training the Support Vector Machine classifier
classifier.fit(X_train, Y_train)


# Evaluation of Model

# Accuracy Score

# In[29]:


# accuracy score on training data
X_train_prediciton = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediciton, Y_train)
print(training_data_accuracy)


# In[30]:


print('Accuracy score of trainig data :', training_data_accuracy)


# In[31]:


# accuracy score on test data
X_test_prediciton = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediciton, Y_test)
print(test_data_accuracy)


# Building a Predcitive system

# In[43]:


input_data= (4,110,92,0,0,37.6,0.191,30)

# changing the input data to numpy array

input_data_as_numpy_array = np.array(input_data)

#reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardized the input data

std_data = scaler.transform(input_data_reshaped)

print(std_data)








# In[44]:


prediction = classifier.predict(std_data)


# In[48]:


print(prediction)

if (prediction == 0):
    print('The person is not diabetic')
else:
    print('the person is diabetic')


# In[ ]:




