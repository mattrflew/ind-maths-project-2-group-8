"""
# =============================================================================
# Module: model1.py
# =============================================================================

This module provides functions for the simulation of Model 1.

"""

# =============================================================================
# Importing Modules
# =============================================================================

# Import all intermediate and initialisation functions
from .functions import *
from .initialise import *
from .params_default import params_default

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# =============================================================================
# Model 1
# =============================================================================

def apply_boundary_conditions(x, y, L):
    '''
    Apply periodic boundary conditions
    Creates wrap-around effect, if crossing bounary enter other side 
    '''
    x = x % L
    y = y % L
    return x, y


def update_positions(x, y, vx, vy, dt, L):
    '''
    Update the positions moving dt in the direction of the velocity
    and applying the boundary conditions
    '''
    
    # update positions
    x += vx*dt
    y += vy*dt
    
    # apply boundary conditions
    # x, y = apply_boundary_conditions(x,y,L)
    return x, y


def get_mean_theta_neighbours(x, y, theta, Rsq, N):
    '''
    Compute the local average direction in a circle of radius R around
    each bird. If there are no neighbours, simply keep same theta.
    '''
    mean_theta = theta
    for bird in range(N):
        neighbours = (x-x[bird])**2+(y-y[bird])**2 < Rsq
        sum_x = np.sum(np.cos(theta[neighbours]))
        sum_y = np.sum(np.sin(theta[neighbours]))
        mean_theta[bird] = np.arctan2(sum_y, sum_x)
    
    return mean_theta

def get_obstacles_within_radius(x_bird, y_bird, theta_bird, x_obstacle, y_obstacle, R_obs, fov_angle):
    
    # Determine if obstacle is in radius
    distances_to_obstacles = np.sqrt((x_obstacle - x_bird) ** 2 + (y_obstacle - y_bird)**2)
    is_in_radius = distances_to_obstacles <= R_obs
    
    # Select only object in radius
    x_obs_in_radius = x_obstacle[is_in_radius]
    y_obs_in_radius = y_obstacle[is_in_radius]
    distances = distances_to_obstacles[is_in_radius]
    
    # Determine if obstacle is in FOV
    
    # Calculate angle to the obstacle    
    delta_x = x_obs_in_radius - x_bird
    delta_y = y_obs_in_radius - y_bird
    angles_to_obstacles = np.arctan2(delta_y, delta_x)
    
    # Difference between obstacle and current direction of bird
    angle_diff = np.abs(angles_to_obstacles - theta_bird)
    # angle_diff = (angles_to_obstacles - theta_bird + np.pi/2) % (2*np.pi) - np.pi
    
    # Filter by field of view
    is_in_fov = angle_diff <= fov_angle
    x_obs_in_radius = x_obs_in_radius[is_in_fov]
    y_obs_in_radius = y_obs_in_radius[is_in_fov]
    distances = distances[is_in_fov]
    
    return x_obs_in_radius, y_obs_in_radius, distances


def update_theta(x, y, theta, Rsq, x_obstacle, y_obstacle, R_obs, eta, N, fov_angle):
    '''
    We will do this per bird, since we need to see if each one is within distance of obstacles or not
    
    '''
    # Initialize new theta array
    theta_new = theta.copy()
    
    # Get the mean theta from the neighbours
    mean_theta = get_mean_theta_neighbours(x, y, theta, Rsq, N)
    
    # If no obstacles:
    if x_obstacle.size == 0 and y_obstacle.size == 0:
        theta_new = add_noise_theta(mean_theta, eta, N)
        return theta_new
    
    # Update theta based on obstacles
    for i in range(N):
        # Determine if obstacles in radius
        x_obs_in_radius, y_obs_in_radius, distances = get_obstacles_within_radius(x[i], y[i], theta_new[i], x_obstacle, y_obstacle, R_obs, fov_angle)

        # avoid division by zero
        distances[distances == 0] = np.finfo(float).eps
        
        # Only continue if there are obstacles in radius
        if np.any(distances):
            
            # Naively avoid obstacle by pointing in opposite direction
            avoidance_vectors = np.array([x[i] - x_obs_in_radius, y[i] - y_obs_in_radius])

            # # Normalise by distance (this should(?) make closer distances more important)
            # avoidance_vectors = avoidance_vectors/distances
            # weights = 1/ (distances + 1e-5)

            # weighted_avoidance_vector = avoidance_vectors * weights
            # Sum up the avoidance vectors to get the net avoidance direction
            net_avoidance_vector = np.sum(avoidance_vectors, axis=0)
            # net_avoidance_vector = np.sum(weighted_avoidance_vector, axis=0)
            net_avoidance_vector = net_avoidance_vector / np.linalg.norm(net_avoidance_vector)
            
            # Get angle of avoidance
            # There were a lot of errors here
            if net_avoidance_vector.size >= 2:
                
                # Get the angle of the net avoidance vector
                avoidance_theta = np.arctan2(net_avoidance_vector[1], net_avoidance_vector[0])

                # Limit the amount a bird can turn in one time step
                max_turn_angle = np.radians(50)  # Maximum allowable turn
                avoidance_theta = np.clip(avoidance_theta, -max_turn_angle, max_turn_angle)
                
                # Calculate weighted average between avoidance theta and mean theta from neighbors
                avoidance_weight = 0.5
                theta_new[i] = (1 - avoidance_weight) * mean_theta[i] + avoidance_weight * avoidance_theta
            
            else:
                # If that didn't work, just go to mean theta
                theta_new[i] = mean_theta[i]
    
        
        # If no obstacle, use theta from neighbours and migration goal
        else:
            theta_new[i] = mean_theta[i]
    
    theta_new = add_noise_theta(theta_new, eta, N)
    
    return theta_new


def update_velocities_original(v0, theta, vx_wind, vy_wind):
    '''
    Update the velocities given theta, assuming a constant speed v0
    '''
    vx = v0 * np.cos(theta) + vx_wind
    vy = v0 * np.sin(theta) + vy_wind
    # print(np.sqrt(vx**2 + vy**2))
    
    return vx, vy


def update_velocities(v0, theta, vx_wind, vy_wind, v_wind_max, alpha=0.5):
    '''
    Update the velocities given theta, assuming a constant speed v0 and wind.
    Account for the fact that a bird will reduce effort with tailwind, alpha
    
    Adjusts a bird's v0 based on how much tailwind it observes, relative to a max tailwind.
    '''
    # Bird direction (unit vector based on theta)
    x_dir = np.cos(theta)
    y_dir = np.sin(theta)
    
    # Determine wind amount along path
    wind_along_path = np.maximum(0, vx_wind * x_dir + vy_wind * y_dir)  # Tailwind only, don't consider negatives

    # Adjust bird's airspeed for tailwind
    adjusted_v0 = v0 * (1 - alpha * (wind_along_path / v_wind_max))
    adjusted_v0 = np.maximum(0, adjusted_v0)  # Ensure non-negative airspeed
    
    # Update velocities (groundspeed)
    vx = adjusted_v0 * np.cos(theta) + vx_wind
    vy = adjusted_v0 * np.sin(theta) + vy_wind
    
    # print(np.sqrt(vx**2 + vy**2))
    
    return vx, vy



def step(x, y, vx, vy, x_obstacle, y_obstacle, L, v0, theta, Rsq, R_obs,  eta, fov_angle, N, dt, v0_wind, v_wind_noise, wind_theta, wind_theta_noise):
    '''
    Compute a step in the dynamics:
    - update the positions
    - compute the new velocities
    '''
    x, y = update_positions(x, y, vx, vy, dt, L)
    theta = update_theta(x, y, theta, Rsq, x_obstacle, y_obstacle, R_obs, eta, N, fov_angle)
    
    vx_wind, vy_wind = wind_constant_with_noise(v0_wind, v_wind_noise, wind_theta, wind_theta_noise)
    
    #vx, vy = update_velocities(v0, theta, vx_wind, vy_wind, v_wind_max=10, alpha=0.5)
    vx, vy = update_velocities_original(v0, theta, vx_wind, vy_wind)
    
    return x, y, vx, vy, vx_wind, vy_wind, theta



def update_quiver(q,x,y,vx,vy):
    '''
    Update a quiver with new position and velocity information
    This is only used for plotting
    '''
    q.set_offsets(np.column_stack([x,y]))
    q.set_UVC(vx,vy)
    
    return q


def run_model1(params, plot = False):

    # If no other parameter class is supplied,
    if params is None:

        # Then use default parameters
        params = params_default() 

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
        beta = params.beta,
        n = params.n
    )

    # Fetch the initial birds in the environment
    x, y, vx, vy, theta, r_effective = initialize_birds(
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
    q = plt.quiver(x,y,vx,vy,  scale=4, angles='xy', scale_units='xy', width=0.005)

    # Set figure parameters
    ax.set(xlim=(0, params.L), ylim=(0, params.L))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Initilise lists to plot later
    vx_wind_list = []
    vy_wind_list = []
    clustering_coefficients = []
    dispersions = []
    
    # Do each step, updating the quiver and plotting the new one
    for i in range(params.Nt):

        x, y, vx, vy, vx_wind, vy_wind, theta = step(
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
        
        # Append metrics
        dispersions.append(calculate_dispersion(x, y))
        clustering_coefficients.append(get_clustering_coefficient(vx, vy, params.v0, vx_wind, vy_wind, params.N))
        
    # Distances to goal at end
    distances_to_goal = calculate_distance_to_goal(x, y, params.goal_x_loc, params.goal_y_loc)
    
    # Number of flocks at end
    num_flocks = number_of_flocks(x, y, r_effective)
    
    return dispersions, distances_to_goal, clustering_coefficients, num_flocks
    