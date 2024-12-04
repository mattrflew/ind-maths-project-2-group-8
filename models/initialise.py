"""
# =============================================================================
# Module: initialise.py
# =============================================================================

This module provides functions for initialising birds and obstacles within the
environment. 

The reason this module is kept separate is because initialisation does not
change across the different models.

For the initialisation of birds we define:

1) initialize_birds_random() - 
2) initialize_birds_square() - 
3) initialize_birds_triangle() -

For the initialision of objects we define:

1) make_rectangular_obstacle() - 
2) make_circular_obstacle() - 
3) make_elliptical_obstacle() - 
3) get_obstacle_rectangle_grid() -

For the addition of wind we define:

1) wind_constant_with_noise()
2) wind_dynamic()
3) wind_spatial()
4) wind_combined(x, y, t, A_x, A_y, k, f)

Master functions:

1) initialize_birds():

   Depending on the method of initialisation chosen, this function returns the 
   x, y, vx, vy, theta parameters of the initialised birds.

2) initialize_obstacles():

   Depending on the shape of obstacles chosen, this function returns the x, y points
   of the objects in the environment.

3) wind():

   Depending on the type of wind chosen, this function returns the 

"""

# =============================================================================
# Importing Modules
# =============================================================================

# Import all intermediate functions
from functions import *

import importlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# =============================================================================
# Initialise Birds Functions
# =============================================================================

# Birds are defined by:
# - Their position (x,y) 
# - Their velocit (vx,vy)

# The magnitudes of all birds' velocities are the same.

def initialize_birds_random(N, L, v0):
    '''
    Set initial positions, direction, and velocities 
    '''
    # Bird positions
    x = np.random.rand(N, 1)*L
    y = np.random.rand(N, 1)*L

    # Bird velocities
    theta = 2 * np.pi * np.random.rand(N, 1)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    return x, y, vx, vy, theta


def initialize_birds_square(N, L, v0, theta_start, eta):
    '''
    Set initial positions as a uniform placement starting at the edge of the box.
    Set direction and velocity to be uniform with a small amount of noise 
    '''
    
    # Set bird initial flock as a square
    N_per_side = int(np.sqrt(N))
    
    # Midpoint of area 
    midpoint = L//2
    
    # Define the x locations (start in centre of perimeter)
    half_length = N_per_side // 2
    start = midpoint - half_length
    x_locs = np.arange(start, start + N_per_side)
    
    # Define the y locations (start from bottom)
    y_locs = np.arange(0, N_per_side)
    
    # Define bird starting points
    # Initialise lists
    x = []
    y = []
    
    for x_loc in x_locs:
        for y_loc in y_locs:
            x.append([x_loc])
            y.append([y_loc])
    
    # Turn into numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Bird Angle
    # Make all birds same starting angle
    theta = np.ones((len(x),1))*theta_start
    
    # Add noise
    theta = add_noise_theta(theta, eta, N)
    
    
    # Bird velocities
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    
    
    return x, y, vx, vy, theta


def initialize_birds_triangle(N, L, v0, theta_start, eta):
    '''
    Set initial positions as a triangle starting at the bottome edge of the simulation box.
    Set direction and velocity to be uniform with a small amount of noise 
    
    Triangle is shaped based on an angle (be an obstuse angle). Make it an obtuse isoceles triangle
    Birds should spaced by 2 m in both the x and y directions
    '''
    min_distance = 2
    top_angle = np.radians(150)

    # Properties of obtuse isoceles triangle
    bottom_angle = (np.pi - top_angle)/2

    # Initialise parameters and lists
    x, y = [], []
    total_birds = 0
    row = 0
    x_row_0 = L // 2

    # Loop through rows of triangle, starting with 1 in first row. 

    while total_birds < N:
        # y position of this row
        y_pos = -row * min_distance

        # base length of triangle in this row
        base_length = 2*abs(y_pos) / np.tan(bottom_angle)
        half_base = base_length/2
        
        # number of birds that can fit in the current row
        birds_in_row = int(base_length / min_distance) + 1

        # limit the number of birds in the row if total exceeds N
        if total_birds + birds_in_row > N:
            birds_in_row = N - total_birds
        
        # add birds around the symmetric line of triangle
        # go from left to right
        x_row_start = x_row_0 - half_base
        
        for i in range(birds_in_row):
            x.append(x_row_start + (i * min_distance))
            y.append(y_pos)
        
        # update total birds and row counters
        total_birds += birds_in_row
        row += 1

    # Turn into numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # shift all y_pos up so birds start within the domain
    y = y + (row-1)*min_distance

    # Bird Angle
    # Make all birds the same starting angle
    theta = np.ones((len(x), 1)) * theta_start

    # Add noise to angle
    theta = add_noise_theta(theta, eta, N)

    # Bird velocities
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    
    return x, y, vx, vy, theta


# =============================================================================
# Initialise Obstacles Functions
# =============================================================================

# Obstacles are defined by:
# - Their position (x,y) 

def make_rectangular_obstacle(x_centre, y_centre, L1, L2, n=25):
    '''
    Returns x,y points defining a rectangular-shaped obstacle
    '''
    
    # Number of points per side
    points_per_side = n // 4
    
    # Half lengths for width and height
    l1 = L1 / 2
    l2 = L2 / 2
    
    # Corners of the rectangle
    top_left = [x_centre - l1, y_centre + l2]
    top_right = [x_centre + l1, y_centre + l2]
    bottom_left = [x_centre - l1, y_centre - l2]
    bottom_right = [x_centre + l1, y_centre - l2]
    
    # Initialize lists for x and y points
    x_points = []
    y_points = []
    
    # Generate points along each side
    # Top edge (left to right)
    x_points.extend(np.linspace(top_left[0], top_right[0], points_per_side))
    y_points.extend([top_left[1]] * points_per_side)
    
    # Right edge (top to bottom)
    x_points.extend([top_right[0]] * points_per_side)
    y_points.extend(np.linspace(top_right[1], bottom_right[1], points_per_side))
    
    # Bottom edge (right to left)
    x_points.extend(np.linspace(bottom_right[0], bottom_left[0], points_per_side))
    y_points.extend([bottom_left[1]] * points_per_side)
    
    # Left edge (bottom to top)
    x_points.extend([bottom_left[0]] * points_per_side)
    y_points.extend(np.linspace(bottom_left[1], top_left[1], points_per_side))
    
    return x_points, y_points

def make_circular_obstacle(x_centre, y_centre, R, n=20):
    '''
    Returns x,y points defining a circular-shaped obstacle
    '''
    angles = np.linspace(0, 2 * np.pi, n)
    
    x = x_centre + R*np.cos(angles)
    y = y_centre + R*np.sin(angles)
    
    return x, y


def make_elliptical_obstacle(x_centre, y_centre, Rx, Ry, n=20):
    '''
    Returns x,y points defining a elliptical-shaped obstacle
    '''
    angles = np.linspace(0, 2 * np.pi, n)
    
    x = x_centre + Rx*np.cos(angles)
    y = y_centre + Ry*np.sin(angles)
    
    return x, y

def get_obstacle_rectangle_grid(L, nrows, ncols, x_spacing, y_spacing, offset, beta):
    '''
    Define a grid of centre points for the obstacles in a rectangle pattern from a pre-specified x and y spacing.
    
    Considers the top half of the domain to be the "wind farm area"
    
    Makes the grid from the centre of this "wind farm area"
    
    x_spacing, y_spacing -> distances between centre points of obstacles
    
    offset -> y offset applied to every second column
    
    beta -> amount of shear applied to angle the grid (in radians). 0 applies no shear.
    '''
    
    # Defining the "wind farm area"
    # in the x direction it goes from 0 to L
    # in the y direction it goes from L/2 to L

    min_x, max_x = L/4, 3*L/4
    min_y, max_y = L/2, L
    
    midpoint_x = min_x + (max_x - min_x)/2
    midpoint_y = min_y + (max_y - min_y)/2
    
    # Define the grid now
    # Centre x and y locations around midlines
    total_x_distance = ncols*x_spacing
    x_locs = np.arange(min_x, min_x + total_x_distance, x_spacing)
    temp_mid = np.mean(x_locs)

    x_locs = x_locs + midpoint_x - temp_mid

    total_y_distance = nrows*y_spacing
    y_locs = np.arange(min_y, min_y + total_y_distance, y_spacing)
    temp_mid = np.mean(y_locs)

    y_locs = y_locs + midpoint_y - temp_mid
    
    # Initialise x and y arrays
    x_centres, y_centres = [], []
    
    # Define the grid locations
    for i, x in enumerate(x_locs):
        for y in y_locs:
            x_centres.append(x)
            
            # Handle the offset
            if not i % 2:
                y_centres.append(y)
            else:
                y_centres.append(y + offset)

    # Apply the shear (shear to the right)
    shear_matrix = np.array([[1, np.tan(beta)],[0, 1]]) # Shear defined here
    grid_points = np.stack([x_centres, y_centres], axis=1) # Combine locations together to apply shear
    
    # Apply the shear
    sheared_grid = np.dot(grid_points, shear_matrix.T)

    # Split back into separate lists
    x_sheared, y_sheared = sheared_grid[:, 0], sheared_grid[:, 1]

    # Move x locations back such it is still centered around the symmetric line
    current_midpoint = (max(x_sheared) + min(x_sheared))/2
    x_offset = midpoint_x - current_midpoint # distance from current centre to desired centre
    x_sheared = x_sheared + x_offset
    
    # Overwrite the variables to return, note that if beta=0 then x_sheared is the same as x_centres, y_sheared is the same as y_centres
    x_centres, y_centres = x_sheared, y_sheared
    return x_centres, y_centres

# =============================================================================
# Wind Functions
# =============================================================================

def wind_constant_with_noise(v0_wind, v_wind_noise, wind_theta):
    '''
    Returns the x, y components of wind based on a constant angle.
    '''
    # Add random noise to the wind
    v0_wind += v_wind_noise * (np.random.rand(1) - 0.5)[0]

    # Get x, y velocity components
    vx_wind = v0_wind * np.cos(wind_theta)
    vy_wind = v0_wind * np.sin(wind_theta)
    
    return vx_wind, vy_wind

def wind_dynamic(t, A_x, A_y, f):
    '''
    Returns the x, y components of wind as sinusoidal functions over time.
    '''
    vx_wind = A_x * np.sin(2 * np.pi * f * t)
    vy_wind = A_y * np.cos(2 * np.pi * f * t)
    
    return vx_wind, vy_wind

def wind_spatial(x, y, A_x, A_y, k):
    '''
    Returns the x, y components of wind as functions of position with exponential decay.
    '''
    vx_wind = A_x * np.exp(-k * x)
    vy_wind = A_y * np.exp(-k * y)

    return vx_wind, vy_wind

def wind_combined(x, y, t, A_x, A_y, k, f):
    '''
    Returns the x, y components of wind as a combination of spatial decay and dynamic sinusoidal variation.
    '''
    vx_spatial = A_x * np.exp(-k * x)
    vy_spatial = A_y * np.exp(-k * y)
    
    vx_dynamic = A_x * np.sin(2 * np.pi * f * t)
    vy_dynamic = A_y * np.cos(2 * np.pi * f * t)
    
    # Combine spatial and dynamic components
    vx_wind = vx_spatial + vx_dynamic
    vy_wind = vy_spatial + vy_dynamic
    
    return vx_wind, vy_wind


# =============================================================================
# Master Functions
# =============================================================================

def initialize_birds(N, L, v0, theta_start, eta, method):
    """
    Master function to initialise birds based on the specified method

    The outputs of each of the individual functions are x, y, vx, vy, theta
    """

    if method == "random":
        return initialize_birds_random(N, L, v0)

    elif method == "uniform":
        return initialize_birds_square(N, L, v0, theta_start, eta)

    elif method == "v-flock":
        return initialize_birds_triangle(N, L, v0, theta_start, eta)

    else:
        raise ValueError(f"Unknown initialisation method: {method}. Choose from 'random', 'uniform', 'v-flock'.")

def initialize_obstacles(L, num_obstacles, nrows, ncols, shape, x_spacing, y_spacing, offset, beta):
    '''
    Master function to initialise obstacles and get lists of their x, y points
    '''
    x_centres, y_centres = get_obstacle_rectangle_grid(L, nrows, ncols, x_spacing, y_spacing, offset, beta)
    
    # Initalise lists
    x_obstacle_list = []
    y_obstacle_list = []
    
    for i in range(num_obstacles):
        
        # Make obstacles depending on specified shape
        if shape == "rectangular":
            x_obs, y_obs = make_rectangular_obstacle(x_centre, y_centre, L1, L2, n=25)

        elif shape == "circular":
            x_obs, y_obs = make_circular_obstacle(x_centre, y_centre, R, n=20)

        elif shape == "elliptical":
            x_obs, y_obs = make_elliptical_obstacle(x_centre, y_centre, Rx, Ry, n=20)

        else:
            raise ValueError(f"Unknown initialisation shape: {shape}. Choose from 'rectangular', 'circular', 'elliptical'.")

        x_obstacle_list.append(x_obs)
        y_obstacle_list.append(y_obs)
    
    # Concatenate lists for analysis
    x_obstacle = np.concatenate(x_obstacle_list)
    y_obstacle = np.concatenate(y_obstacle_list)
    
    return x_obstacle_list, y_obstacle_list, x_obstacle, y_obstacle

def wind(x, y, t, v0_wind, v_wind_noise, wind_theta, A_x, A_y, k, f, method):
    """
    Master function to add wind to the model, based on the specified method

    The outputs of each of the individual functions are vx_wind, vy_wind
    """

    if method == "constant":
        return wind_constant_with_noise(v0_wind, v_wind_noise, wind_theta)

    elif method == "dynamic":
        return wind_dynamic(t, A_x, A_y, f)

    elif method == "spatial":
        return wind_spatial(x, y, A_x, A_y, k)

    elif method == "combined":
        return wind_combined(x, y, t, A_x, A_y, k, f)
    else:
        raise ValueError(f"Unknown initialisation method: {method}. Choose from 'constant', 'dynamic', 'spatial', 'combined'.")


