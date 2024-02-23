# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MANOGARAN S
RegisterNumber:212223240081
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/

```

## Output:
DATA SET
![Screenshot 2024-02-23 101600](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/f17b5eb3-7b2b-4d51-af03-85fdf209a0d0)

HEAD VALUE
![Screenshot 2024-02-23 101609](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/d5bc2093-66b2-4240-a9e9-1f3aeb7e4735)

TAIL VALUE
![Screenshot 2024-02-23 101615](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/aa3cf29b-ca08-4f7c-9733-006af374df82)

X AND Y VALUE
![Screenshot 2024-02-23 101643](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/e3559555-1c5b-41bb-9fb4-e694787622f0)

PREDICTION VALUE OF X AND Y
![Screenshot 2024-02-23 101700](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/c26f5496-8d7a-4ad2-840d-bfe5cb5d53d5)

MSE,MAE AND RMSE
![Screenshot 2024-02-23 101732](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/23381278-8972-4a10-af4a-218821467f8e)

TRAINING SET
![Screenshot 2024-02-23 101744](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/8b665ae2-d824-4fc7-8fdb-2ea370411125)

TESTING SET
![Screenshot 2024-02-23 101753](https://github.com/manogarans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/139331782/b1cece80-b3d5-4648-879d-e6e145ad53b3)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
