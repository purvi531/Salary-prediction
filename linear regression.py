# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:09:34 2021

@author: HP
"""
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("salaryData.csv");
print(dataset)

dataset.head(5) #give first 5 rows of data
dataset.columns #seeing name of columns
dataset.shape #give no. of rows and colounms
dataset.info() #type of dataset
dataset.min() #give minimum value
dataset.describe() #mean,max,min
dataset.iloc[2:6]  #give subset of data in between these indeces
dataset.nunique() #give no of unique value

#drawing data
#x-year of experience(feature)
#y-salary(predict it with the help of x)
x=dataset.iloc[:,:-1].values #except the last taking all coloumns and we want 2d array 
#feature should always be in 2d shape
y=dataset.iloc[:,1].values #it is a 1d array (taking last coloumn)
plt.xlabel('experince')
plt.ylabel('salary')
plt.scatter(x,y,color='red',marker='+')

#dividing into training and testing into 80:20
#for it we use sklearn

import sklearn
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=1)

#1/3 data is in test remaining 2/3 is in train
#random_state=1 i, it will always pick same data set for training and testing
#(1 is not compalsory, we can give any int value here)


#creating simple linear model
from sklearn.linear_model import LinearRegression
model=LinearRegression() #y=ax+b
#fitting data into model
model.fit(xtrain,ytrain)
#so my model is ready
#now predict
y_pred=model.predict(xtest)
print(y_pred)
print(ytest)
#differece of y_pred(predicted y) and ytest(actual)

#y=ax+b
model.coef_  #give value of a
model.intercept_ #give value of b
#so predict someone having expreience of x=9 years
model.predict([[9]])

#vizualization
plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,model.predict(xtrain)) #for drawing line
plt.show()