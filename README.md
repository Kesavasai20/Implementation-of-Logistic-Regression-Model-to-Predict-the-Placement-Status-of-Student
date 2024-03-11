# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values
## Program:
```py
'''
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: K KESAVA SAI
RegisterNumber:  212223230105
'''
import pandas as pd
data=pd.read_csv('placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

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

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Placement data :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/0b35ed69-2808-4543-aabf-03efa747b385)
## Salary data :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/a0f1b10c-6733-4fb2-a224-43f9f75faa44)
## Checking the null() function :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/0ad4b4bf-d6c8-4482-b25d-ff28b21b62f3)
## Data Duplicate :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/3dc75727-5997-4820-b0cc-2a8f2dcf7631)
## Print data :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/0c304a2a-beba-4961-8d46-a4c174261f32)
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/25c4b950-2608-4a84-9e7e-8818d4d1326c)
## Data-status :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/9a2b7555-1b7d-4c34-8df3-be09a6270a10)
## y_prediction array :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/f07bca7c-1daf-4e46-bf41-b0fafefc411f)
## Accuracy value :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/20a2f103-0927-485e-98d6-549cba3f25aa)
## Confusion array :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/cd364d15-f672-4f80-8113-886a7b1e3260)
## Classification report :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/5fff909c-f581-4162-a725-a4838ae672ad)
## Prediction of LR :
![image](https://github.com/Kesavasai20/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/138849303/34ab3b34-7d96-4326-9973-c183c62a45b4)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
