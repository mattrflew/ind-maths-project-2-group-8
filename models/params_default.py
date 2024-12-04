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
        
        # Bird speed
        self.v0 = 17.38               # Speed of birds (constant)
        
        # -------------------------------------------------------------------------
        # Time & Space
        # -------------------------------------------------------------------------
        self.L = 7000                 # Size of box (L*L = Area of a wind farm)
        self.dt = 0.2                 # Time step
        self.T = self.L/self.v0       # Minimum simulation length (seconds)
        self.Nt = int(self.T/self.dt) # No. of time steps

        # -------------------------------------------------------------------------
        # Birds
        # -------------------------------------------------------------------------
        self.vmax = 30.0              # Maximum velocity (?)
        self.eta = 0.5                # maximum random fluctuation in angle (in radians)
        self.R_bird = 2               # Interaction radius (bird-bird)
        self.Rsq = self.R_bird**2     # Square of the interaction radius
        self.N = 250                  # number of birds
        self.fov_angle = np.pi        # Field of View of birds
        self.R = 2                    # A distance over which birds can observe their neighbours, R,
        self.r_min = 0.1              # a minimum distance they would like to maintain, r.
        self.R_obs = 0.5              # Interaction radius (bird - obstacles)
        
        # Migratory goal vector
        self.goal_x = 1               # X-component of common goal direction vector
        self.goal_y = 1               # Y-component of common goal direction vector

        # Flock shape
        self.theta_start = np.pi / 2  # Starting bird direction
        self.bird_method = "v-flock"  # The method of bird initialisation

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
        # Obstacle creation
        self.n = 30                   # Number of points that defines the obstacle
        
        # Physical parameters
        self.diameter = 100           # Diameter of turbine
        self.width = 10               # Width of turbine at widest point (including nacelle)
        self.shape = "elliptical"     # Default shape is elliptical       
        self.Rx = self.diameter/2     # Elliptical radius in x direction
        self.Ry = self.width/2        # Elliptical radius in y direction
        
        # Grid layout
        self.nrows = 3                # Number of rows in grid
        self.ncols = 10               # Number of columns in grid
        self.num_obstacles = (        # Total number of obstacles in grid
            self.nrows * self.ncols
        )      

        # Spacing of grid
        self.rotor_spacing_side = 5                                 # Num diameters to the side between turbines
        self.rotor_spacing_behind = 9                               # Num diameters behind between turbines
        self.x_spacing = (self.rotor_spacing_side + 1)*self.diameter      # Spacing to the side between turbines
        self.y_spacing = (self.rotor_spacing_behind + 1)*self.diameter    # Spacing behind between the turbines
        
        # By having these at 0, it is a rectangular grid
        self.offset = 0                                             # Offset in grid layout, applied to every other row
        self.beta = np.radians(0)                                   # Shear amount in grid layout
        

        # -------------------------------------------------------------------------
        # Wind
        # -------------------------------------------------------------------------
        self.v0_wind = 0                           # Velocity of wind (constant)
        self.v_wind_noise = 0                      # Maximum random fluctuation in wind velocity (in same units as v0_wind)
        self.wind_theta = np.pi/2                  # Wind direction (in radians)
        self.wind_theta_noise = 0                  # Maximum random fluctuation in wind angle (in radians)
        self.wind_method = "constant"              # Options: "constant", "dynamic", "spatial", "combined"
        self.f = 0.05                              # Frequency of wind variation
        self.A_x = 1.0                             # Amplitude of wind in x direction
        self.A_y = 1.0                             # Amplitude of wind in y direction
        self.k = 0.1                               # Decay rate for spatial wind variation