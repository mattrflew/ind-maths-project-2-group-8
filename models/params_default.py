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

class params_default():
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
    def __init__(

        self,

        # -----------------------------------------------------------------------------
        # Time & Space
        # -----------------------------------------------------------------------------

        dt = 0.2,                 # Time step
        Nt = 100,                 # No. of time steps
        L = 10,                   # Size of box (Area of a wind farm)

        # -----------------------------------------------------------------------------
        # Birds
        # -----------------------------------------------------------------------------

        v0 = 1.0,                 # velocity of birds (constant)
        vmax = 2.0,               # Maximum velocity 
        eta = 0.5,                # maximum random fluctuation in angle (in radians)
        R_bird = 1,               # interaction radius (bird-bird)
        Rsq = R_bird**2,          # square of the interaction radius
        N = 25,                   # number of birds
        fov_angle = np.pi,        # Field of View of birds
        R =  1,                   # A distance over which birds can observe their neighbours, R,
        r_min =  0.1,             # a minimum distance they would like to maintain, r.
        R_obs = 0.5,              # Interaction radius (bird - obstacles)

        # Migratory goal vector
        goal_x = 1,          
        goal_y = 1,

        theta_start = np.pi/2, 
        bird_method = "v-flock",

        # -----------------------------------------------------------------------------
        # 'Mixing' parameters
        # -----------------------------------------------------------------------------

        # These are weights for the different contributions to the bird's velocity:

        lam_c = .1,              # Centering weight
        lam_a = .15,             # Avoidance weight
        lam_m = .1,              # Matching weight
        lam_o = .0,              # Obstacle weight
        lam_g = .05,             # Migratory weight
        lam_w = .1,              # Wind weight

        # -----------------------------------------------------------------------------
        # Obstacles
        # -----------------------------------------------------------------------------

        num_obstacles = 8,
        nrows = 4,
        ncols = 2,
        shape ="elliptical",
        x_spacing = 15
        y_spacing = 10,
        offset= 5, 
        beta = np.radians(0),

        # -----------------------------------------------------------------------------
        # Wind
        # -----------------------------------------------------------------------------

        v0_wind = 0.5,                          # Velocity of wind (constant)
        v_wind_noise = 0.1,                     # Maximum random fluctuation in wind velocity (in same units as v0_wind)
        wind_theta = 0,                         # Wind direction 
        wind_method = "combined"                # Options: "constant", "dynamic", "spatial", "combined"

        f = 0.05,                               # Frequency of wind variation
        A_x = 1.0,                              # Amplitude of wind in x direction
        A_y = 1.0,                              # Amplitude of wind in y direction
        k = 0.1,                                # Decay rate for spatial wind variation

    ):

        # -------------------------------------------------------------------------
        # Define Time & Space
        # -------------------------------------------------------------------------
        self.dt = dt
        self.Nt = Nt
        self.L = L

        # -------------------------------------------------------------------------
        # Define Birds
        # -------------------------------------------------------------------------
        self.v0 = v0
        self.vmax = vmax
        self.eta = eta
        self.R_bird = R_bird
        self.Rsq = R_bird ** 2  # Derived value
        self.N = N
        self.fov_angle = fov_angle
        self.R = R
        self.r_min = r_min
        self.R_obs = R_obs

        # Migratory goal vector
        self.goal_x = goal_x
        self.goal_y = goal_y

        self.theta_start = theta_start
        self.method = method

        # -------------------------------------------------------------------------
        # Define 'Mixing' Parameters
        # -------------------------------------------------------------------------
        self.lam_c = lam_c
        self.lam_a = lam_a
        self.lam_m = lam_m
        self.lam_o = lam_o

        # -------------------------------------------------------------------------
        # Define Obstacles
        # -------------------------------------------------------------------------
        self.num_obstacles = num_obstacles
        self.nrows = nrows
        self.ncols = ncols
        self.shape = shape
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.offset = offset
        self.beta = beta

        # -------------------------------------------------------------------------
        # Define Wind
        # -------------------------------------------------------------------------
        self.v0_wind = v0_wind
        self.v_wind_noise = v_wind_noise
        self.wind_theta = wind_theta