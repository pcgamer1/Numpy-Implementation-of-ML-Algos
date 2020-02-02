# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 13:00:28 2018

@author: Sarthak
"""

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```
Defining the function

```python
def sig(x,deriv=False):
    if deriv==True:
        return (x*(1-x))

    return 1/(1+np.exp(-x))
   
```

Initializing

```python
x=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y=np.array([[0],[1],[1],[0]])

#seed
np.random.seed(1)

#synapses
syn0=2*np.random.random((3,4)) - 1
syn1=2*np.random.random((4,1)) - 1

```
Running Forward-Propagation and Back-Propagation

```python
#predicting values
for i in range(50000):
    l0=x
    z=l0.dot(syn0)
    l1=sig(l0.dot(syn0))
    l2=sig(l1.dot(syn1))
    if i%5000==0:
        print(l2)
 
    #Backpropagation
    l2_error=y-l2
    l2_delta=l2_error*sig(l2,deriv=True)
    l1_error=l2_delta.dot(syn1.T)
    l1_delta=l1_error*sig(l1,deriv=True)
    
    #Updating Synapses
    syn1+=l1.T.dot(l2_delta)
    syn0+=l0.T.dot(l1_delta)
 
#Output
print(l2)
```
