#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
data=pd.read_csv('placement_Data.csv')
data.head()


# In[5]:


data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()


# In[6]:


data1.isnull().sum()


# In[7]:


data1.duplicated().sum()


# In[8]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1


# In[15]:


x=data1.iloc[:,:-1]
x


# In[16]:


y=data1["status"]
y


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[19]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred


# In[20]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[26]:


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion


# In[23]:


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


# In[24]:


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


# In[ ]:




