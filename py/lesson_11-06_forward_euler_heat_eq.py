# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 11:25:10 2019

Implementation of a forward Euler scheme for the heat equation as explained by Wen Shen in Lesson 11-06.

URL: https://www.youtube.com/watch?v=LgJ2i2BW9ok&list=PLbxFfU5GKZz3D4NPYvvY7dvXiZ0awd4zn&index=120

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

x_min = 0.0
x_max = 1.0
n = 31 # number of nodes along the domain
h = (x_max - x_min) / (n-1)
dt = 0.45 * h**2 # timestemp in seconds
y_0 = 0 # value at the left (lower) boundary of the domain
y_n = 0 # value at the right (upper) boundary of the domain
#f = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-0.5 * (x**2))
f = lambda x: np.sin(2*np.pi*x)

def get_node_row(i, h, n, dt):
    """
    Function to get a row of the matrix corresponding to the node i.

    i: the index of the node whose row we are computing
    h: the distance between nodes along the domain
    n: the number of nodes along the domain
    dt: the timestep
    """

    assert i>0 and i<(n-1), "Node i is not an interior node. (i = " + str(i)+ ", n = " + str(n) +")"

    g = dt / (h**2)

    if i == 1:
        left_part = np.asarray([1-2*g, g])
        right_part = np.zeros(n-4)
        row = np.concatenate((left_part, right_part))
    elif i == (n-2):
        left_part = np.zeros(i-2)
        right_part = np.asarray([g, 1-2*g])
        row = np.concatenate((left_part, right_part))
    else:
        left_part = np.zeros(i-2)
        central_part = np.asarray([g, 1-2*g, g])
        right_part = np.zeros((n-3)-i)
        row = np.concatenate((left_part, central_part, right_part))

    return row

def get_y_t0(f, x, y_0, y_n):
    y = f(x[1:(len(x)-1)]) # the initial value of y
    y = np.insert(y, 0, y_0)
    y = np.append(y, y_n)
    return y

def next_y(A, y, y_0, y_n):
    y_next = A.dot(y[1: (len(y)-1) ])
    y_next[0] += y_0
    y_next[len(y_next)-1] += y_n
    y_next = np.insert(y_next, 0, y_0)
    y = np.append(y_next, y_n)

    return y

rows = [get_node_row(i, h, n, dt) for i in range(1, n-1)]
A = np.vstack(rows)


x = np.linspace(x_min, x_max, n)
y = get_y_t0(f, x, y_0, y_n)
plt.plot(x, y, label = "t = 0")
rows = [get_node_row(i, h, n, dt) for i in range(1, n-1)]
A = np.vstack(rows)
times_to_plot = [0, 1, 2, 3, 4, 5, 10, 50, 100, 150, 200]
for k in range(1,201):
    y = next_y(A, y, y_0, y_n)
    if k in times_to_plot:
        plt.plot(x, y, label = "t = " + str(round(k*dt, 4)))

plt.legend(loc = "best")