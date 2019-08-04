# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:19:37 2019

Implementation of example of the concepts discussed by Wen Shen in Lesson 10-7, "Neumann Boundary Condition, Poisson's Equation"

Poisson's Equation: y'' = f(x)

boundary conditions: y'(0) = a, y(1) = b, where a an b are defined below

url: https://www.youtube.com/watch?v=c8oXqxYQ-qk&list=PLbxFfU5GKZz3D4NPYvvY7dvXiZ0awd4zn&index=113

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

x_min = 0.0
x_max = 10.0
n = 30
h = (x_max - x_min) / n
a = 3
b = 2

x = np.linspace(x_min, x_max, num = n)
f = np.sin(x[:len(x)-1])

import numpy as np

def get_inner_row(i, n, h):
    """
    Get a row of the matrix corresponding to a node at which we're solving for the value of f.

    i: the index of the node
    n: the number of nodes in the domain
    h: the distance between nodes
    """

    assert i >=0 and i<(n-1), "Node i is not a node we are solving for. (i = " + str(i)+ ", n = " + str(n) +")"

    if i == 0:
        left_part = np.asarray([-2,2])
        right_part = np.zeros((n-1)-2)
        row = np.concatenate((left_part, right_part))
    elif i == n-2:
        left_part = np.zeros(i-1)
        right_part = np.asarray([1,-2])
        row = np.concatenate((left_part, right_part))
    else:
        left_part = np.zeros(i-1)
        right_part = np.zeros((n-2) - (i+1))
        central_part = np.asarray([1, -2, 1])
        row = np.concatenate((left_part, central_part, right_part))

    return row

def get_w(n, h, f, a, b):
    w = (h**2) * f
    w[0] += 2*h*a
    w[len(w)-1] -= b

    return w

def get_y(A, w, b):
    A_inv = np.linalg.inv(A)
    y = A_inv.dot(w)
    y = np.append(y, b)

    return y


rows = [get_inner_row(i, n, h) for i in range(n-1)]
A = np.vstack(rows)

# an array of values for the left boundary condition
lbcs = np.linspace(-3, 3, 7)
plt.figure()
for lbc in lbcs:
    w = get_w(n, h, f, lbc, b)
    y = get_y(A, w, b)
    plt.plot(x, y, marker = 'o', label = "a = " + str(lbc) + ", b = " + str(b))

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Approximate solution for y'' = sin(x). Varying LBC")
plt.legend(loc = 'best')

# an array of values for the right boundary condition
rbcs = np.linspace(0, 5, 6)
plt.figure()
for rbc in rbcs:
    w = get_w(n, h, f, a, rbc)
    y = get_y(A, w, rbc)
    plt.plot(x, y, marker = 'o', label = "a = " + str(a) + ", b = " + str(rbc))

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Approximate solution for y'' = sin(x). Varying RBC")
plt.legend(loc = 'best')

# an array of functions for the RHS of the laplace equation
RHS_fs = {"sin": np.sin, "exp": np.exp, "sinh": np.sinh}
plt.figure()
for key in RHS_fs.keys():
    RHS_f = RHS_fs[key]
    f_vals = RHS_f(x[:len(x)-1])
    w = get_w(n, h, f_vals, a, b)
    y = get_y(A, w, b)
    plt.plot(x, y, marker = 'o', label = "RHS: " + key + "(x)")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Approximate solution for y'' = f(x). Varying RHS Function")
plt.legend(loc = 'best')

