"""
# =============================================================================
# Module: params_default.py
# =============================================================================

This module provides a class of default parameters, that are passed to each 
model if no other class of parameters is defined.

"""
# =============================================================================
# Importing Modules
# =============================================================================

import numpy as np

# =============================================================================
# Default Parameters
# =============================================================================

class params_default:
    """
    Defines the parameters for our model

    Inputs:
    Parameters related to:
    - Time and space
    - Birds 
    - Velocity mixing
    - Obstacles 
    - Wind

    Output:
        parameters Object
            Containing all the defined parameters
    """

    def __init__(self):

        # -------------------------------------------------------------------------
        # Time & Space
        # -------------------------------------------------------------------------
        self.dt = 0.2                 # Time step
        self.Nt = 100                 # No. of time steps
        self.L = 10                   # Size of box (Area of a wind farm)

        # -------------------------------------------------------------------------
        # Birds
        # -------------------------------------------------------------------------
        self.v0 = 1.0                 # velocity of birds (constant)
        self.vmax = 2.0               # Maximum velocity 
        self.eta = 0.5                # maximum random fluctuation in angle (in radians)
        self.R_bird = 1               # Interaction radius (bird-bird)
        self.Rsq = self.R_bird**2     # Square of the interaction radius
        self.N = 25                   # number of birds
        self.fov_angle = np.pi        # Field of View of birds
        self.R = 1                    # A distance over which birds can observe their neighbours, R,
        self.r_min = 0.1              # a minimum distance they would like to maintain, r.
        self.R_obs = 0.5              # Interaction radius (bird - obstacles)

        # Migratory goal vector
        self.goal_x = 1          
        self.goal_y = 1

        self.theta_start = np.pi / 2 
        self.bird_method = "v-flock"

        # -------------------------------------------------------------------------
        # 'Mixing' parameters
        # -------------------------------------------------------------------------
        self.lam_c = 0.1              # Centering weight
        self.lam_a = 0.15             # Avoidance weight
        self.lam_m = 0.1              # Matching weight
        self.lam_o = 0.0              # Obstacle weight
        self.lam_g = 0.05             # Migratory weight
        self.lam_w = 0.1              # Wind weight

        # -------------------------------------------------------------------------
        # Obstacles
        # -------------------------------------------------------------------------
        self.num_obstacles = 8
        self.nrows = 4
        self.ncols = 2
        self.shape = "elliptical"
        self.x_spacing = 15
        self.y_spacing = 10
        self.offset = 5
        self.beta = np.radians(0)

        # -------------------------------------------------------------------------
        # Wind
        # -------------------------------------------------------------------------
        self.v0_wind = 0.5                           # Velocity of wind (constant)
        self.v_wind_noise = 0.1                      # Maximum random fluctuation in wind velocity (in same units as v0_wind)
        self.wind_theta = 0                          # Wind direction 
        self.wind_method = "combined"                # Options: "constant", "dynamic", "spatial", "combined"

        self.f = 0.05                                # Frequency of wind variation
        self.A_x = 1.0                               # Amplitude of wind in x direction
        self.A_y = 1.0                               # Amplitude of wind in y direction
        self.k = 0.1                                 # Decay rate for spatial wind variation