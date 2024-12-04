"""
# =============================================================================
# Module: functions.py
# =============================================================================

This module provides all the intermediate functions that are used within each 
model, that are not specific to the chosen model.

"""

# =============================================================================
# Importing Modules
# =============================================================================

# Import all intermediate and initialisation functions
from .functions import *
from .initialise import *

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# =============================================================================
# Helper Functions
# =============================================================================

def add_noise_theta(theta, eta, N):
    '''
    Update theta with a random amount of noise between -eta/2 and eta/2
    '''
    theta += eta * (np.random.rand(N, 1) - 0.5)
    
    return theta

# -----------------------------------------------------------------------------
# Vector Operations
# -----------------------------------------------------------------------------

def normalise(v):
    """ 
    Normalise a vector to length 1    
    """
    norm = np.linalg.norm(v)
    
    if norm == 0:
        return v
    
    norm_v = v / norm
    
    return norm_v

def distance(u, v):
    """
    Calculate the Euclidean distance between two vectors.

    Inputs:
        u (numpy array): First vector.
        v (numpy array): Second vector.

    Outputs:
        distance: The Euclidean distance between the two vectors.
    """
    distance = np.linalg.norm(u - v)
    
    return distance

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------

def update_quiver(q,x,y,vx,vy):
    '''
    Update a quiver with new position and velocity information
    This is only used for plotting
    '''
    q.set_offsets(np.column_stack([x,y]))

    q.set_UVC(vx,vy)
    
    return q

# -----------------------------------------------------------------------------
# Metrics Calculations
# -----------------------------------------------------------------------------

def calculate_dispersion(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    distances = np.sqrt((x - x_mean)**2 + (y - y_mean)**2)
    dispersion = np.mean(distances)
    return dispersion

def calculate_path_offset(x, y, goal_x, goal_y):
    center_x, center_y = np.mean(x), np.mean(y)
    offset = np.sqrt((center_x - goal_x)**2 + (center_y - goal_y)**2)
    return offset

def get_clustering_coefficient(vx, vy, v0, vx_wind, vy_wind, N):
    
    # Sum the vx and vy components (with wind included)
    sum_terms_x = np.sum(vx)
    sum_terms_y = np.sum(vy)
    sum_terms = np.linalg.norm([sum_terms_x, sum_terms_y])
    
    # Get the expected v, v0 + v_wind
    # Expected total velocity magnitude per bird
    # v_expected = np.sqrt(v0**2 + vx_wind**2 + vy_wind**2) # doesn't work?
    
    # Average bird speed (calculate directly)
    v_expected = np.mean(np.sqrt(vx**2 + vy**2))
    
    # Calculate coefficient
    clustering_coefficient = (1/(N*v_expected))*sum_terms
    
    return clustering_coefficient

def return_metric_statistics(dispersion_values, offset_values, clustering_coefficients):
    '''
    Returns the averaged metrics (i.e. results of the simulation)
    '''
    avg_dispersion = np.mean(dispersion_values)
    avg_offset = np.mean(offset_values)
    avg_clustering_coefficient = np.mean(clustering_coefficients)
    
    return avg_dispersion, avg_offset, avg_clustering_coefficient


# -----------------------------------------------------------------------------
# Simulation Function
# -----------------------------------------------------------------------------

def run_simulation(model, params, plot = False):

    # If no other parameter class is supplied,
    if params is None:

        # Then use default parameters
        params = params_default() 

    if model == "model1":

    elif model == "Model2":

    elif model == "Model3":

    # Fetch the obstacles in the environment
    x_obstacle_list, y_obstacle_list, x_obstacle, y_obstacle = initialize_obstacles(
        L = params.L,
        num_obstacles = params.num_obstacles,
        nrows = params.nrows, 
        ncols = params.ncols, 
        shape = params.shape, 
        Rx = params.Rx, 
        Ry = params.Ry, 
        x_spacing = params.x_spacing, 
        y_spacing = params.y_spacing, 
        offset = params.offset, 
        beta = params.beta
    )

    # Fetch the initial birds in the environment
    x, y, vx, vy, theta = initialize_birds(
        N = params.N, 
        L = params.L, 
        v0 = params.v0, 
        theta_start = params.theta_start, 
        eta = params.eta,
        method = params.bird_method
    )

    # Set up a figure
    fig, ax = plt.subplots(figsize = (8,8))

    # Plot obstacle(s) - Plot the "list" to visualise the different obstaclces properly
    for xx, yy in zip(x_obstacle_list, y_obstacle_list):
        ax.plot(xx, yy, 'r-')

    # Plot initial quivers
    q = plt.quiver(x,y,vx,vy)

    # Set figure parameters
    ax.set(xlim=(0, params.L), ylim=(0, params.L))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Initilise lists to plot later
    vx_wind_list = []
    vy_wind_list = []
    clustering_coefficients = []

    # Do each step, updating the quiver and plotting the new one
    for i in range(params.Nt):

        x, y, vx, vy, vx_wind, vy_wind = step(
            x = x, 
            y = y, 
            vx = vx, 
            vy = vy, 
            x_obstacle = x_obstacle, 
            y_obstacle = y_obstacle, 
            L = params.L, 
            v0 = params.v0, 
            theta = theta, 
            Rsq = params.Rsq, 
            R_obs = params.R_obs,  
            eta = params.eta, 
            fov_angle = params.fov_angle, 
            N = params.N, 
            dt = params.dt, 
            v0_wind = params.v0_wind, 
            v_wind_noise = params.v_wind_noise, 
            wind_theta = params.wind_theta, 
            wind_theta_noise = params.wind_theta_noise
        )

        if plot:
            q = update_quiver(q, x, y, vx, vy)
            clear_output(wait=True)
            display(fig)
        
        # Append wind information
        vx_wind_list.append(vx_wind)    
        vy_wind_list.append(vy_wind)
        
        # Append clustering coefficient
        clustering_coefficients.append(get_clustering_coefficient(vx, vy, params.v0, vx_wind, vy_wind, params.N))
