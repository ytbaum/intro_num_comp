# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:32:28 2019

Implementation of Crank-Nicolson Scheme on the heat equation as explained by Wen Shen in Lesson 11-9.

url: https://www.youtube.com/watch?v=FS4VHlLwXVc&list=PLbxFfU5GKZz3D4NPYvvY7dvXiZ0awd4zn&index=123

@author: user
"""

import numpy as np

x_min = 0.0
x_max = 1.0
n = 31 # number of nodes along the domain
h = (x_max - x_min) / (n-1)
dt = 1 # timestemp in seconds
y_0 = 0 # value at the left (lower) boundary of the domain
y_n = 0 # value at the right (upper) boundary of the domain
#f = lambda x: 1/np.sqrt(2*np.pi) * np.exp(-0.5 * (x**2))
f = lambda x: np.sin(2*np.pi*x)

def get_node_row_LHS(i, h, n, dt):
    """
    Function to generate the row corresponding to node i in the LHS matrix.

    i: the index of the node that this row corresponds to
    h: the distance between nodes along the domain
    n: the number of nodes along the domain

    # NOTE: this matrix's shape should be (n-2) x (n-2)
    """

    assert i>0 and i<(n-1), "Node i is not an interior node. (i = " + str(i)+ ", n = " + str(n) +")"

    r = dt / (2*(h**2))

    if i == 1:
        left_part = np.asarray([1+(2*r), -r])
        right_part = np.zeros(n-4)
        row = np.concatenate((left_part, right_part))
    elif i == (n-2):
        left_part = np.zeros(i-2)
        right_part = np.asarray([-r, 1+(2*r)])
        row = np.concatenate((left_part, right_part))
    else:
        center_part = np.asarray([-r, 1+(2*r), -r])
        left_part = np.zeros(i-2)
        right_part = np.zeros((n-3)-i)
        row = np.concatenate((left_part, center_part, right_part))

    return row

def get_node_row_RHS(i, h, n, dt):
    """
    Function to generate the row corresponding to node i in the matrix that transforms y_t into w, the RHS of the equation to solve for y_t+1.

    i: the index of the node that this row corresponds to
    h: the distance between nodes along the domain
    n: the number of nodes along the domain
    """

    assert i>0 and i<(n-1), "Node i is not an interior node. (i = " + str(i)+ ", n = " + str(n) +")"

    r = dt / (2*(h**2))

    if i == 1:
        left_part = np.asarray([1-(2*r), r])
        right_part = np.zeros(n-4)
        row = np.concatenate((left_part, right_part))
    elif i == (n-2):
        left_part = np.zeros(i-2)
        right_part = np.asarray([r, 1-(2*r)])
        row = np.concatenate((left_part, right_part))
    else:
        center_part = np.asarray([r, 1-(2*r), r])
        left_part = np.zeros(i-2)
        right_part = np.zeros((n-3)-i)
        row = np.concatenate((left_part, center_part, right_part))

    return row

def get_w(y, h, n, dt, y_0, y_n):
    """
    Function to compute the RHS of the matrix equation for computing y_t+1 at the next time value

    y: the values of y at the current time value, i.e. y_t
    h: the distance between nodes along the domain
    """

    rows = [get_node_row_RHS(i, h, n, dt) for i in range(1, n-1)]
    RHS_mat = np.vstack(rows)

    ### TO DO: I SHOULDN'T HAVE TO DO THIS STEP ###
    ### MEANS SOMETHING IS WRONG WITH THE WAY I'M GENERATING RHS_MAT ###
    RHS_mat = -RHS_mat

    w = RHS_mat.dot(y[1:(len(y)-1)])

    r = dt / (2*(h**2))
    w[0] += (r*y_0)
    w[len(w)-1] += (r*y_n)

    return w

def next_y(A, w, y_0, y_n):
    A_inv = np.linalg.inv(A)
    y = A_inv.dot(w)
    y = np.insert(y, 0, y_0)
    y = np.append(y, y_n)

    return y

def get_y_t0(f, x, y_0, y_n):
    y = f(x[1:(len(x)-1)]) # the initial value of y
    y = np.insert(y, 0, y_0)
    y = np.append(y, y_n)
    return y

x = np.linspace(x_min, x_max, n)
y = get_y_t0(f, x, y_0, y_n)
plt.plot(x, y, label = "t = 0")
A = [get_node_row_LHS(i, h, n, dt) for i in range(1, n-1)]
times_to_plot = [0, 1, 2, 3, 4, 5, 10, 50, 100, 150, 200]
for k in range(1,201):
    w = get_w(y, h, n, dt, y_0, y_n)
    y = next_y(A, w, y_0, y_n)
    if k in times_to_plot:
        plt.plot(x, y, label = "t = " + str(k))

plt.legend(loc = "best")

