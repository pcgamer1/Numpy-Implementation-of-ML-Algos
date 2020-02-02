# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 21:25:49 2018

@author: Sarthak
"""
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
Importing Data and performing required operations

```python
data=pd.read_csv("Salary_Data.csv")
x=data["YearsExperience"].values
y=data["Salary"].values
theta=[0,0]
x1=np.ones(30)
m=len(x)
xf=np.array([x1,x])
xft=np.transpose(xf)
cost=np.sum((np.dot(xft,theta)-y)**2)/(2*m)
print(cost)

```
Gradient Descent

```python
def grad_descent(x,y,m,theta,alpha,numit):
    x_trans=np.transpose(xft)
    for i in range(numit):
        h=np.dot(xft,theta)
        loss=h-y
        gradient=np.dot(x_trans,loss)/m
        theta=theta - alpha*gradient
        if i%10000==0:
            print(gradient)
    return theta
 ```
Calling the function

```python
alpha=0.001
numit=100000
theta=grad_descent(x,y,m,theta,alpha,numit)
print(theta)
y_predict= np.dot(xft,theta)

```
