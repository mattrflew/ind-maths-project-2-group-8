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

# Import all initialisation functions
from .initialise import *

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from sklearn.cluster import DBSCAN

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

def number_nearest_neightbours(x, y, Rsq, N):
    '''
    Return the number of nearest neighbours. If a flock is more disperesed, we would expect the average number of nearest neighbours 
    to be lower.
    '''
    num_neighbours = []
    
    for bird in range(N):
        neighbours = (x-x[bird])**2+(y-y[bird])**2 < Rsq
        num_neighbours.append(sum(neighbours))
    
    return num_neighbours


def number_of_flocks(x, y, r_effective):

    positions = np.column_stack((x,y))

    # Apply DBSCAN
    clustering = DBSCAN(eps=r_effective, min_samples=1).fit(positions)

    # Number of unique clusters
    num_flocks = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

    return num_flocks



def return_metric_statistics(dispersion_values, offset_values, clustering_coefficients, num_neighbours):
    '''
    Returns the averaged metrics (i.e. results of the simulation)
    '''
    avg_dispersion = np.mean(dispersion_values)
    avg_offset = np.mean(offset_values)
    avg_clustering_coefficient = np.mean(clustering_coefficients)
    avg_num_neighbours = np.mean(num_neighbours)
    
    return avg_dispersion, avg_offset, avg_clustering_coefficient, avg_num_neighbours


# -----------------------------------------------------------------------------
# Simulation Function
# -----------------------------------------------------------------------------


