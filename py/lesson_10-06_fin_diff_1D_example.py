#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:48:04 2019

Code accompanying lesson 10-6: Finite Difference Method in 1-D. Examples.

url: https://www.youtube.com/watch?v=XE65rn9db2I&list=PLbxFfU5GKZz3D4NPYvvY7dvXiZ0awd4zn&index=112

@author: yoni
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sin

def get_inner_node_row(i, h, n):
    """
    Function to get a row of the matrix that solves for the values of y on one of the inner nodes (not touching the boundary) of the domain.

    i: the index of the node this row is solving for
    h: the distance between nodes
    n: the total number of nodes in the domain
    """

    assert i>0 and i<(n-1), "Node i is not an interior node. (i = " + str(i)+ ", n = " + str(n) +")"

    if i == 1:
        left_part = np.asarray([-2+(4*h**2), 1])
        right_part = np.zeros((n-3)-i)
        row = np.concatenate((left_part, right_part))
    elif i == n-2:
        left_part = np.zeros(i-2)
        right_part = np.asarray([1, -2+(4*h**2)])
        row = np.concatenate((left_part, right_part))
    else:
        central_part = np.asarray([1, -2+(4*h**2), 1])
        left_part = np.zeros(i-2)
        right_part = np.zeros((n-3)-i)
        row = np.concatenate((left_part, central_part, right_part))

    return row

def get_b(x, y_0, y_n, h):
    """
    Function to get the load vector b for the equation Ay = b

    x: the vector of x values at nodes along the domain
    y_0: the boundary value of y at the left side of the domain
    y_n: the boundary value of y at the right side of the domain
    """

    b = 4*(h**2) * x[1:(len(x)-1)] # only use values of x at non-boundary nodes
    b[0] -= y_0
    b[len(b)-1] -= y_n

    return b

x_min = 0.0
x_max = 1.0
n = 11
h = (x_max - x_min) / (n-1)
y_0 = 0
y_n = 2

x = np.linspace(x_min, x_max, num = n)

rows = [get_inner_node_row(i, h, n) for i in range(1, n-1)]
A = np.vstack(rows)
b = get_b(x, y_0, y_n, h)
A_inv = np.linalg.inv(A)
y_interior = A_inv.dot(b)
y = np.insert(y_interior, 0, y_0)
y = np.append(y, y_n)

x_analytical = np.linspace(x_min, x_max, 100)
y_analytical = (1/sin(2)) * np.sin(2*x_analytical) + x_analytical
plt.plot(x_analytical, y_analytical, label = "Analytical Solution")
plt.scatter(x, y, color = 'orange', label = "Approx. Solution")
plt.legend(loc = 'best')
plt.title(r"Solution of y'' = -4(y-x)")
plt.xlabel("X")
plt.ylabel("Y")


