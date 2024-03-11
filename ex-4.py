#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FuelConsumption.csv')
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission')
plt.show()


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FuelConsumption.csv')
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='blue', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.xlabel('Cylinders/Engine Size')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission and Engine Size vs CO2 Emission')
plt.legend()
plt.show()


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('FuelConsumption.csv')
plt.scatter(df['CYLINDERS'], df['CO2EMISSIONS'], color='green', label='Cylinder')
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='red', label='Engine Size')
plt.scatter(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'], color='yellow', label='Fuel Consumption')
plt.xlabel('Cylinders/Engine Size/Fuel Consumption')
plt.ylabel('CO2 Emission')
plt.title('Cylinder vs CO2 Emission, Engine Size vs CO2 Emission, and Fuel Consumption vs CO2 Emission')
plt.legend()
plt.show()


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
X_train_cylinder, X_test_cylinder, y_train_cylinder, y_test_cylinder = train_test_split(X_cylinder, y_co2, test_size=0.2, random_state=42)
model_cylinder = LinearRegression()
model_cylinder.fit(X_train_cylinder, y_train_cylinder)


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('FuelConsumption.csv')
X_fuel = df[['FUELCONSUMPTION_COMB']]
y_co2 = df['CO2EMISSIONS']
X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_co2, test_size=0.2, random_state=42)
model_fuel = LinearRegression()
model_fuel.fit(X_train_fuel, y_train_fuel)


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('FuelConsumption.csv')
X_cylinder = df[['CYLINDERS']]
y_co2 = df['CO2EMISSIONS']
ratios = [0.1, 0.4, 0.5, 0.8]
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X_cylinder, y_co2, test_size=ratio, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Train-Test Ratio: {1-ratio}:{ratio} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')


# In[ ]:


ex-4


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




