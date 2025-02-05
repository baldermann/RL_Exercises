# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 01:09:01 2024

@author: aunko
"""

import numpy as np
import torch


"""
    Calculate Function with PyTorch
"""

a = torch.Tensor([2.0])
a.requires_grad = True

b = torch.Tensor([1.0])
b.requires_grad = True

def linear_model(x, a, b):
    
    return a * x + b

y = linear_model(torch.Tensor([4.0]), a, b)

print(y)   # AddBackwards.... Berechnungshistorie mit Ableitungskoeffizienten speichert

print(y.grad_fn)



with torch.no_grad():
    
    y2 = linear_model(torch.Tensor([4]), a, b)
    
print(y2) # Kein AddBackwards.... Berechnungshistorie wurde nicht gespeichert. Backpropagation könnte nicht mehr ausgeführt werden.

print(y2.grad_fn)



"""
    --------- Partielle Ableitung mit PyTorch
"""

import torch

def linear_model(x, a, b):
    
    return a * x + b

a = torch.Tensor([20.0])
a.requires_grad = True

b = torch.Tensor([10.0])
b.requires_grad = True

y = linear_model(torch.Tensor([4.0]), a, b)
    
y.backward()
print(a.grad)
print(b.grad)
    


    
