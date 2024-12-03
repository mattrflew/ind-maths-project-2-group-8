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

import numpy as np


# =============================================================================
# Helper Functions
# =============================================================================

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

def add_noise_theta(theta, eta, N):
    '''
    Update theta with a random amount of noise between -eta/2 and eta/2
    '''
    theta += eta * (np.random.rand(N, 1) - 0.5)
    
    return theta

def update_quiver(q,x,y,vx,vy):
    '''
    Update a quiver with new position and velocity information
    This is only used for plotting
    '''
    q.set_offsets(np.column_stack([x,y]))

    q.set_UVC(vx,vy)
    
    return q
