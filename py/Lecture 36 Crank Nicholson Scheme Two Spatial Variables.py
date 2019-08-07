# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:32:21 2019

Implementation of the Alternating Direction Implicit Method for
2D Crank-Nicholson Scheme.

As explained in lecture by Sunil Kumar, Dept. of Physics, IIT Madras, India,
in "Lecture 36 - The Crank - Nicholson Scheme For Two Spatial Variables"

url: https://www.youtube.com/watch?v=yCwFBeUk1_E

I will use a 2D heat equation as an example on which to implement this method.

Much of this code, notably the code for generating the LHS and RHS matrices, is copy/pasted from
lesson_11_09_crank_nicholson_heat_eq.py.

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation

x_min = -100.0
x_max = 100.0
y_min = -100.0
y_max = 100.0
n = 9 # number of nodes along the domain
num_interior_nodes = (n-2) * (n-2)
h = (x_max - x_min) / (n-1) # we will have the same distance between nodes along the x and y directions
num_levels = 20 # number of levels to divide the contour plot into
num_frames = 25 # number of frames to include in the animation
dt = 5 # timestemp in seconds

f = lambda x, y: 1/np.sqrt(2*np.pi) * np.exp(-0.5 * ((x**2)+(y**2))) # initial condition
x = np.linspace(x_min, x_max, n)
y = np.linspace(y_min, y_max, n)
x_grid, y_grid = np.meshgrid(x, y)
z_t0 = f(x_grid, y_grid)

plt.figure()
qcs = plt.contourf(x_grid, y_grid, z_t0, num_levels, cmap = "Reds")
levels = qcs.levels.copy()
plt.title("Initial value of the target function z(x,y)")
plt.xlabel("X")
plt.ylabel("Y")
cb = plt.colorbar()

# boundary conditions
left_bound = z_t0[x_grid == x.min()]
right_bound = z_t0[x_grid == x.max()]
top_bound = z_t0[y_grid == y.max()]
bottom_bound = z_t0[y_grid == y.min()]

# corresponds to lambda in the lectures of Prof. Kumar
# used in constructing the matrices for LHS and RHS of the iteration equation

lam = dt/(h**2)

def get_RHS_mat(num_nodes):
    """
    Function to get the matrix to compute from vector z_t the RHS of the iteration equation.
    """
    main_diag = np.repeat(1-(4*lam), num_nodes)
    first_sub_diag = np.repeat(2*lam, num_nodes-1)
    RHS_mat = diags([first_sub_diag, main_diag, first_sub_diag], [-1, 0, 1])

    return RHS_mat

def get_LHS_mat(num_nodes):
    """
    Function to construct the matrix for the LHS of the iteration equation.

    """
    main_diag = np.repeat(1+(4*lam), num_nodes)
    first_sub_diag = np.repeat(-2*lam, num_nodes-1)
    LHS_mat = diags([first_sub_diag, main_diag, first_sub_diag], [-1, 0, 1])

    return LHS_mat

def get_w_t(z, n, RHS_mat):
    """
    Construct w, the vector on the RHS of the iteration equation, for the first of the two alternating directions. I.e. where we are computing (d^2)z/dy^2 at t+(1/2) based on (d^2)z/dx^2 at t.
    """

    interior_nodes = z[1:(n-1), 1:(n-1)].copy() # this is a square array of shape (n-2) x (n-2)
    v = interior_nodes.flatten()
    w = RHS_mat.dot(v)
    w_sq = w.reshape((n-2, n-2))

    # add boundary conditions on left and right boundaries (since we are handling (d^2)z/dx^2 at this stage)
    w_sq[:,0] += 2*lam * left_bound[1:(n-1)]
    w_sq[:,(n-3)] += 2*lam * right_bound[1:(n-1)]
    w = w_sq.flatten()

    return w

def get_w_t_half(z, n, RHS_mat):
    """
    Construct w, the vector on the RHS of the iteration equation, for the second of the two alternating directions. I.e. where we are computing (d^2)z/dx^2 at t+1 based on (d^2)z/dy^2 at t+(1/2).
    """

    interior_nodes = z[1:(n-1), 1:(n-1)].copy() # this is a square array of shape (n-2) x (n-2)
    v = interior_nodes.flatten()
    w = RHS_mat.dot(v)
    w_sq = w.reshape((n-2), (n-2))

    # # add boundary conditions on bottom and top boundaries (since we are handling (d^2)z/dy^2 at this stage)
    w_sq[0, :] += 2*lam * bottom_bound[1:(n-1)]
    w_sq[(n-3), :] += 2*lam * top_bound[1:(n-1)]

    # in keeping with the "Alternating direction" idea, transpose the interior_nodes array
    w_sq = w_sq.T

    w = w_sq.flatten()
    return w

LHS_mat = get_LHS_mat(num_interior_nodes)
LHS_inv = np.linalg.inv(LHS_mat.toarray())
RHS_mat = get_RHS_mat(num_interior_nodes)
z = z_t0.copy()

### Generate Animated Gif ###

fig, ax = plt.subplots()
fig.set_tight_layout(True)
print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))

def update(i):

    global z, n, RHS_mat, LHS_inv, x, x_grid, y, y_grid
    # first of the two "alternating directions"

    if i == 0:
        z = z_t0.copy()

    w_t = get_w_t(z, n, RHS_mat)
    interior_nodes = LHS_inv.dot(w_t)

    z = np.empty((n,n))
    z[1:(n-1), 1:(n-1)] = interior_nodes.reshape((n-2, n-2))

    # impose boundary conditions
    z[x_grid == x.min()] = left_bound
    z[x_grid == x.max()] = right_bound
    z[y_grid == y.max()] = top_bound
    z[y_grid == y.min()] = bottom_bound

    # second of the two "alternating directions

    w_t_half = get_w_t_half(z, n, RHS_mat)
    interior_nodes = LHS_inv.dot(w_t_half)

    z = np.empty((n,n))
    z[1:(n-1), 1:(n-1)] = interior_nodes.T.reshape((n-2, n-2))

    # impose boundary conditions
    z[x_grid == x.min()] = left_bound
    z[x_grid == x.max()] = right_bound
    z[y_grid == y.max()] = top_bound
    z[y_grid == y.min()] = bottom_bound

    # Generate image for this frame of the gif #

    label = 'timestep {0}'.format(i)
    print(label)

    # apparently the colorbar is stored as another axes. in this case it will be the last axes under the current figure, since it is the last thing that was plotted.
    # remove it so that it's no longer there when I add an updated colorbar for the updated contour plot
    for a in fig.axes:
        a.remove()

    c_plot = plt.contourf(x_grid, y_grid, z, levels, cmap = "Reds")
    plt.title("Value of the target function z(x,y) at t=" + str(i*dt))
    plt.xlabel("X")
    plt.ylabel("Y")
    fig.colorbar(c_plot)

    return ax

anim = FuncAnimation(fig, update, frames = np.arange(num_frames), interval = 750)
#plt.show()
gif_name = "C:/Users/user/Documents/test/test_gif.gif"
anim.save(gif_name)