# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:33:01 2018

@author: Sarthak
"""

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
Defining required functions

```python
def sigmoid(z):
    val=1/(1+np.exp(-z))
    return val

def logcalc(y): 
    val=np.log(y)
    return val
```
Importing the Data and performing required operations

```python
data=pd.read_csv("titanic.csv")
data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)
male=pd.get_dummies(data["Sex"],drop_first=True)
Pclass=pd.get_dummies(data["Pclass"],drop_first=True)
Embark=pd.get_dummies(data["Embarked"],drop_first=True)
data=pd.concat([data,male,Pclass,Embark],axis=1)
data.drop(["Sex","Pclass","Embarked"],axis=1,inplace=True)
data.dropna(inplace=True)
    
x=data.drop(["Survived"],axis=1)
x=x.values
y=data["Survived"].values
demn=np.ones([714,1])
xf=np.hstack([demn,x])
xft=np.transpose(xf)
```    
Gradient Descent

```python
def graddescent(x,y,numit,theta,alpha,m):
    xft
    for i in range(numit):
        hi=np.dot(xf,theta)
        h=sigmoid(hi)
        loss=h-y
        gradient=np.dot(xft,loss)/m
        theta=theta - alpha * gradient
        if i%10000==0 :
            print(theta)
    return theta
```
Calling the functions

```python
alpha=0.01
theta=[0,0,0,0,0,0,0,0,0,0]
numit=1000000
m=len(y)
theta=graddescent(x,y,numit,theta,alpha,m)
print(theta)
hi=np.dot(xf,theta)
onem=np.ones(714)
#cost=np.sum(-y*logcalc(sigmoid(hi)) + (onem-y)*logcalc(onem-sigmoid(hi)
cost=np.sum((sigmoid(hi)-y) ** 2)/(2 * m)
print(cost)
ho=sigmoid(hi)

```
