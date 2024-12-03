"""
# =============================================================================
# Module: params_default.py
# =============================================================================

This module provides a class of default parameters, that are passed to each 
model if no other class of parameters is defined.

"""

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

        # Migratory goal vector
        goal_x = 1          
        goal_y = 1

        # -----------------------------------------------------------------------------
        # 'Mixing' parameters
        # -----------------------------------------------------------------------------

        # These are weights for the different contributions to the bird's velocity:

        lam_c = 1.0,              # Centering Parameter
        lam_a = 1.0,              # Avoidance Parameter
        lam_m = 1.0,              # Matching Parameter
        lam_o = 1.0,              # Obstacle Parameter

        # -----------------------------------------------------------------------------
        # Obstacles
        # -----------------------------------------------------------------------------

        R_obs = 0.5              # Interaction radius (bird - obstacles)

        # -----------------------------------------------------------------------------
        # Wind
        # -----------------------------------------------------------------------------

        v0_wind = 0.5           # Velocity of wind (constant)
        v_wind_noise = 0.1      # Maximum random fluctuation in wind velocity (in same units as v0_wind)
        wind_theta = 0          # Wind direction 

        ):

        # Time and space
        self.dt = dt
        self.Nt = Nt
        self.L = L

        # Birds
        self.v0 = v0
        self.vmax = vmax
        self.eta = eta
        self.R_bird = R_bird
        self.Rsq = self.R_bird ** 2  # Derived value
        self.N = N
        self.fov_angle = fov_angle
        self.R = R
        self.r_min = r_min

        # Mixing Parameters
        self.lam_c = lam_c
        self.lam_a = lam_a
        self.lam_m = lam_m
        self.lam_o = lam_o

        # Obstacles
        self.R_obs = R_obs

        # Wind
        self.v0_wind = v0_wind
        self.v_wind_noise = v_wind_noise
        self.wind_theta = wind_theta