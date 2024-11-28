# =============================================================================
# Importing Packages
# =============================================================================  

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# =============================================================================
# Parameters
# =============================================================================

# -----------------------------------------------------------------------------
# Birds
# -----------------------------------------------------------------------------

v0 = 1.0                 # velocity of birds (constant)
vmax = 2.0               # Maximum velocity 
eta = 0.5                # maximum random fluctuation in angle (in radians)
R_bird = 1               # interaction radius (bird-bird)
Rsq = R_bird**2          # square of the interaction radius
N = 25                   # number of birds
fov_angle = np.pi        # Field of View of birds
R =  1                   # A distance over which birds can observe their neighbours, R,
r_min =  0.1             # a minimum distance they would like to maintain, r.

# -----------------------------------------------------------------------------
# 'Mixing' parameters
# -----------------------------------------------------------------------------

# These are weights for the different contributions to the bird's velocity
lam_c = 1.0              # Centering Parameter
lam_a = 1.0              # Avoidance Parameter
lam_m = 1.0              # Matching Parameter
lam_o = 1.0              # Obstacle Parameter

# -----------------------------------------------------------------------------
# Time & Space
# -----------------------------------------------------------------------------

dt = 0.2                 # Time step
Nt = 100                 # No. of time steps
L = 10                   # Size of box (Area of a wind farm)

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



