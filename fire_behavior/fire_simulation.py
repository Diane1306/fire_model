#!/usr/bin/env python
# coding: utf-8

"""
This is a model for 2D wind-driven fire simulation in different conditions. Except that wind_fire_simulation() is simulated in cellular automata, all are simulated in analytic ways.

The main reference paper for the wind_fire_simulation() is: Alexandridis, A., D. Vakalis, C.I. Siettos, and G.V. Bafas. “A Cellular Automata Model for Forest Fire Spread Prediction: The Case of the Wildfire That Swept through Spetses Island in 1990.” Applied Mathematics and Computation 204, no. 1 (October 2008): 191–201. 
"""

import numpy as np
import matplotlib.pylab as plt
import sys
import math
import random
import copy
from matplotlib import colors as c

def show_animation(delay=.01):
    """Animate function to show animated figures. By Dirk Colbry."""
    from IPython.display import display, clear_output
    import time
    fig = plt.gcf()
    # Sleep for half a second to slow down the animation
    time.sleep(delay)
    clear_output(wait=True);  # Clear output for dynamic display
    display(fig);            # Reset display
    fig.clear();             # Prevent overlapping and layered plots


def homofire_nowind():
    """
    Simulate fire on homogeneous field with no wind: fire spread rate is (sqrt{n+1} - sqrt{n})*r.
    
    Black shows burned area, red shows burning area, and white shows unburned area.
    """

    r_homo = [1]
    n = 1
    r = 1
    while r > .1:
        r = np.sqrt(n+1) - np.sqrt(n)
        r_homo.append(r)
        n = n + 1

    radius = np.array(r_homo).cumsum()

    for i in range(len(r_homo)-1):
        """Red color shows burning areas and black color represents burned areas."""
        if i == 0:
            circle = plt.Circle((0, 0), radius[i], color='red')
            ax = plt.gca()
            ax.set_xlim((-6, 6))
            ax.set_ylim((-6, 6))
            ax.axis('square')
            ax.add_artist(circle)
        else:
            circle1 = plt.Circle((0, 0), radius[i], color='black')
            circle2 = plt.Circle((0, 0), radius[i+1], color='red')
            ax = plt.gca()
            ax.set_xlim((-6, 6))
            ax.set_ylim((-6, 6))
            ax.axis('square')
            ax.add_artist(circle2)
            ax.add_artist(circle1)
        show_animation()


def homofire_cwind():
    """
    Simulate fire on homogeneous field with constant wind.
    
    Black shows burned area, red shows burning area, and white shows unburned area.
    """

    r_homo = [1]
    n = 1
    r = 1
    while r > .1:
        r = np.sqrt(n+1) - np.sqrt(n)
        r_homo.append(r)
        n = n + 1

    radius = np.array(r_homo).cumsum()

    def xy(t):
        theta = np.linspace(0, 2*np.pi, 100)
        a = radius[t] / t
        g = 1.0
        f = 2.0
        h = 1.0
        x = a * t * (f * np.cos(theta) + g)
        y = a * t * h * np.sin(theta)
        return x, y

    for t in range(1, 20):
        if t == 1:
            x1, y1 = xy(t)
            plt.fill(x1, y1, 'r')
            plt.axis('square')
            plt.xlim(-5, 15)
            plt.ylim(-5, 5)
        else:
            x1, y1 = xy(t)
            x2, y2 = xy(t+1)
            plt.fill(x2, y2, 'r')
            plt.fill(x1, y1, 'k')
            plt.axis('square')
            plt.xlim(-5, 15)
            plt.ylim(-5, 5)
        show_animation()


def homofire_dwind():
    """Simulate fire on homogeneous field with constant wind which has a direction change with time."""

    r_homo = [1]
    n = 1
    r = 1
    while r > .1:
        r = np.sqrt(n+1) - np.sqrt(n)
        r_homo.append(r)
        n = n + 1

    radius = np.array(r_homo).cumsum()

    pi = np.pi
    cos = np.cos
    sin = np.sin

    theta = np.linspace(0, 2*pi, 100)
    a = 1.0
    g = 1.0
    f = 2.0
    h = 1.0
    t0 = 1.0
    x0 = a * t0 * (f * cos(theta) + g)
    y0 = a * t0 * h * sin(theta)

    def xy(x0, y0, t):
        theta = np.linspace(0, 2*pi, 100)
        beta = (pi / 6) * np.log(t)
        a = radius[t] / t
        g = 1.0
        f = 2.0
        h = 1.0
        p = h * cos(beta) * cos(theta) + f * sin(beta) * sin(theta)
        q = h * sin(beta) * cos(theta) - f * cos(beta) * sin(theta)
        r = (p * p * f * f + q * q * h * h) ** (-.5)
        x = x0 + a * t * (g * cos(beta) + r * (p * f*f *
                                               cos(beta) + q * h*h * sin(beta)))
        y = y0 + a * t * (g * sin(beta) + r * (p * f*f *
                                               sin(beta) - q * h*h * cos(beta)))
        return x, y

    for t in range(1, 20):
        if t == 1:
            plt.fill(x0, y0, 'r')
            plt.axis('square')
            plt.xlim(-8, 15)
            plt.ylim(-8, 15)
        else:
            x1, y1 = xy(x0, y0, t)
            x2, y2 = xy(x0, y0, t+1)
            plt.fill(x2, y2, 'r')
            plt.fill(x1, y1, 'k')
            plt.axis('square')
            plt.xlim(-8, 15)
            plt.ylim(-8, 15)
        show_animation()


# 2-D wind-driven fire simulation

def wind_fire_simulation(wd):
    """Simulate wind-driven fire. Wind directions are needed: 'N', 'S', 'E', 'W', 'NW', 'NE', 'SE', 'SW'."""

    # initialize environment

    # number of rows and columns of grid
    n_row = 300
    n_col = 300
    generation = 100

    # the input wind direction decides the theta matrix used later. It's the angle between wind direction and fire spread direction.
    if wd == 'S':
        thetas = [[45, 0, 45],
                  [90, 0, 90],
                  [135, 180, 135]]
    elif wd == 'N':
        thetas = [[135, 180, 135],
                  [90, 0, 90],
                  [45, 0, 45]]
    elif wd == 'W':
        thetas = [[45, 90, 135],
                  [0, 0, 180],
                  [45, 90, 135]]
    elif wd == 'E':
        thetas = [[135, 90, 45],
                  [180, 0, 0],
                  [135, 90, 45]]
    elif wd == 'NW':
        thetas = [[90, 135, 180],
                  [45, 0, 135],
                  [0, 45, 90]]
    elif wd == 'SW':
        thetas = [[0, 45, 90],
                  [45, 0, 135],
                  [90, 135, 180]]
    elif wd == 'SE':
        thetas = [[90, 45, 0],
                  [135, 0, 45],
                  [180, 135, 90]]
    elif wd == 'NE':
        thetas = [[180, 135, 90],
                  [135, 0, 45],
                  [90, 45, 0]]
    else:
        raise ValueError("""Not a valid wind direction for this simulation
Options include N,NE,E,SE,S,SW,W,NW""") 

    def init_vegetation():
        """Initiate vegetation array."""
        veg_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        for i in range(n_row):
            for j in range(n_col):
                veg_matrix[i][j] = 1
        return veg_matrix

    def init_density():
        """Initiate density array."""
        den_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        for i in range(n_row):
            for j in range(n_col):
                den_matrix[i][j] = 1
        return den_matrix

    def init_altitude():
        """Initiate altitude array."""
        alt_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        for i in range(n_row):
            for j in range(n_col):
                alt_matrix[i][j] = 1
        return alt_matrix

    def init_forest():
        """Initiate forest array."""
        forest = [[1 for col in range(n_col)] for row in range(n_row)]
        ignite_col = int(n_col//2)
        ignite_row = int(n_row//2)
        for row in range(ignite_row-2, ignite_row+2):
            for col in range(ignite_col-2, ignite_col+2):
                forest[row][col] = 2
        return forest

    def tg(x):
        """Calculate tangent(x) and convert into degrees."""
        return math.degrees(math.atan(x))

    def get_slope(altitude_matrix):
        """Calculate slope from altitude."""
        slope_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
        for row in range(n_row):
            for col in range(n_col):
                sub_slope_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                if row == 0 or row == n_row-1 or col == 0 or col == n_col-1:  # margin is flat
                    slope_matrix[row][col] = sub_slope_matrix
                    continue
                current_altitude = altitude_matrix[row][col]
                sub_slope_matrix[0][0] = tg(
                    (current_altitude - altitude_matrix[row-1][col-1])/1.414)
                sub_slope_matrix[0][1] = tg(
                    current_altitude - altitude_matrix[row-1][col])
                sub_slope_matrix[0][2] = tg(
                    (current_altitude - altitude_matrix[row-1][col+1])/1.414)
                sub_slope_matrix[1][0] = tg(
                    current_altitude - altitude_matrix[row][col-1])
                sub_slope_matrix[1][1] = 0
                sub_slope_matrix[1][2] = tg(
                    current_altitude - altitude_matrix[row][col+1])
                sub_slope_matrix[2][0] = tg(
                    (current_altitude - altitude_matrix[row+1][col-1])/1.414)
                sub_slope_matrix[2][1] = tg(
                    current_altitude - altitude_matrix[row+1][col])
                sub_slope_matrix[2][2] = tg(
                    (current_altitude - altitude_matrix[row+1][col+1])/1.414)
                slope_matrix[row][col] = sub_slope_matrix
        return slope_matrix

    def calc_pw(theta):
        """Calculate wind contribution for each cell."""
        c_1 = 0.045
        c_2 = 0.131
        V = 10
        t = math.radians(theta)
        ft = math.exp(V*c_2*(math.cos(t)-1))
        return math.exp(c_1*V)*ft

    def get_wind(thetas):
        """Get the wind matrix based on the neighbor cells."""
        wind_matrix = [[0 for col in [0, 1, 2]] for row in [0, 1, 2]]

        for row in [0, 1, 2]:
            for col in [0, 1, 2]:
                wind_matrix[row][col] = calc_pw(thetas[row][col])
        wind_matrix[1][1] = 0
        return wind_matrix

    def burn_or_not_burn(abs_row, abs_col, neighbour_matrix):
        """Determine if the cell is going to burn or not."""
        p_veg = {1: -0.3, 2: 0, 3: 0.4}[vegetation_matrix[abs_row][abs_col]]
        p_den = {1: -0.4, 2: 0, 3: 0.3}[density_matrix[abs_row][abs_col]]
        p_h = 0.58
        a = 0.078

        for row in [0, 1, 2]:
            for col in [0, 1, 2]:
                # we only care there is a neighbour that is burning
                if neighbour_matrix[row][col] == 2:
                    # print(row,col)
                    slope = slope_matrix[abs_row][abs_col][row][col]
                    p_slope = math.exp(a * slope)
                    p_wind = wind_matrix[row][col]
                    p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_wind * p_slope
                    if p_burn > random.random():
                        return 2  # start burning

        return 1  # not burning

    def update_forest(old_forest):
        """Update the forest status with loop."""
        result_forest = [[1 for i in range(n_col)] for j in range(n_row)]
        for row in range(1, n_row-1):
            for col in range(1, n_col-1):

                if old_forest[row][col] == 3:
                    # burnt down
                    result_forest[row][col] = old_forest[row][col]
                if old_forest[row][col] == 2:
                    if random.random() < 0.4:
                        result_forest[row][col] = 2
                    else:
                        result_forest[row][col] = 3
                if old_forest[row][col] == 1:
                    neighbours = [[row_vec[col_vec] for col_vec in range(col-1, col+2)]
                                  for row_vec in old_forest[row-1:row+2]]
                    # print(neighbours)
                    result_forest[row][col] = burn_or_not_burn(
                        row, col, neighbours)

        return result_forest

    # start simulation
    vegetation_matrix = init_vegetation()
    density_matrix = init_density()
    altitude_matrix = init_altitude()
    wind_matrix = get_wind(thetas)
    new_forest = init_forest()
    slope_matrix = get_slope(altitude_matrix)

    cMap = c.ListedColormap(['g', 'r', 'k'])
    for k in range(generation):
        new_forest = copy.deepcopy(update_forest(new_forest))
        forest_array = np.array(new_forest)
        #plt.figure(figsize=(10, 8))
        plt.pcolormesh(forest_array, cmap=cMap)
        plt.colorbar(ticks=[1, 2, 3])
        show_animation()
    
    return "Simulation succeed!!"
