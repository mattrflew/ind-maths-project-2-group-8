"""
# =============================================================================
# Module: model1.py
# =============================================================================

This module provides functions for the simulation of Model 1.

"""

# =============================================================================
# Importing Modules
# =============================================================================

# Import all intermediate functions
from functions import *

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
    x, y = apply_boundary_conditions(x,y,L)
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

def add_noise_theta(theta, eta, N):
    '''
    Update theta with a random amount of noise between -eta/2 and eta/2
    '''
    theta += eta * (np.random.rand(N, 1) - 0.5)
    
    return theta

def update_theta(x, y, theta, Rsq, x_obstacle, y_obstacle, R_obs, eta, N, fov_angle):
    '''
    We will do this per bird, since we need to see if each one is within distance of obstacles or not
    
    '''
    # Initialize new theta array
    theta_new = theta.copy()
    
    # Get the mean theta from the neighbours
    mean_theta = get_mean_theta_neighbours(x, y, theta, Rsq, N)
    
    # Update theta based on obstacles
    for i in range(N):
        # Determine if obstacles in radius
        x_obs_in_radius, y_obs_in_radius, distances = get_obstacles_within_radius(x[i], y[i], theta_new[i], x_obstacle, y_obstacle, R_obs, fov_angle)

        # Only continue if there are obstacles in radius
        if np.any(distances):
            
            # Naively avoid obstacle by pointing in opposite direction
            avoidance_vectors = np.array([x[i] - x_obs_in_radius, y[i] - y_obs_in_radius])

            # # Normalise by distance (this should(?) make closer distances more important)
            avoidance_vectors = avoidance_vectors/distances

            # Sum up the avoidance vectors to get the net avoidance direction
            net_avoidance_vector = np.sum(avoidance_vectors, axis=0)
            
            
            # Get angle of avoidance
            # There were a lot of errors here
            if net_avoidance_vector.size >= 2:
                
                # Get the angle of the net avoidance vector
                avoidance_theta = np.arctan2(net_avoidance_vector[1], net_avoidance_vector[0])

                # Calculate weighted average between avoidance theta and mean theta from neighbors
                avoidance_weight = 0.9
                theta_new[i] = (1 - avoidance_weight) * mean_theta[i] + avoidance_weight * avoidance_theta
            
            else:
                # If that didn't work, just go to mean theta
                theta_new[i] = mean_theta[i]
    
        
        # If no obstacle, use theta from neighbours
        else:
            theta_new[i] = mean_theta[i]
        
    
    theta_new = add_noise_theta(theta_new, eta, N)
    
    return theta_new

def update_velocities(v0, theta):
    '''
    Update the velocities given theta, assuming a constant speed v0
    '''
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    return vx, vy

def step(x, y, vx, vy, theta, Rsq, x_obstacle, y_obstacle, eta, fov_angle, N, dt):
    '''
    Compute a step in the dynamics:
    - update the positions
    - compute the new velocities
    '''
    x, y = update_positions(x, y, vx, vy, dt, L)
    theta = update_theta(x, y, theta, Rsq, x_obstacle, y_obstacle, R_obs, eta, N, fov_angle)
    vx, vy = update_velocities(v0, theta)
    
    return x, y, vx, vy

def update_quiver(q,x,y,vx,vy):
    '''
    Update a quiver with new position and velocity information
    This is only used for plotting
    '''
    q.set_offsets(np.column_stack([x,y]))
    q.set_UVC(vx,vy)
    
    return q

def model1_plot():

    # Set up a figure
    fig, ax = plt.subplots(figsize = (8,8))

    # Get the obstacle(s) coordinates
    num_obstacles = 4
    x_obstacle_list = []
    y_obstacle_list = []

    x_centres, y_centres = get_obstacle_centre_grid(L, num_obstacles, nrows=2, ncols=2)
    # x_centres = [2.5, 7.5, 2.5, 7.5]
    # y_centres = [2.5, 2.5, 7.5, 7.5]

    for i in range(num_obstacles):
        x_obs, y_obs = make_circular_obstacle(x_centres[i], y_centres[i], 0.25)

        # x_obs, y_obs = make_rectangular_obstacle(x_centres[i], y_centres[i], 1, 0.2)
        # make_rectangular_obstacle(x_centre, y_centre, L1, L2, n=25)

        x_obstacle_list.append(x_obs)
        y_obstacle_list.append(y_obs)

    # Concatenate lists for analysis
    x_obstacle = np.concatenate(x_obstacle_list)
    y_obstacle = np.concatenate(y_obstacle_list)

    # x_obstacle, y_obstacle = make_rectangular_obstacle(L//2, L//2, 0.5, 0.5)

    # Plot obstacle(s) - Plot the "list" to visualise the different obstaclces properly
    for xx, yy in zip(x_obstacle_list, y_obstacle_list):
        ax.plot(xx, yy, 'r-')

    # Get the initial configuration
    x, y, vx, vy, theta = initialize_birds_random(N, L)

    # Plot initial quivers
    q = plt.quiver(x,y,vx,vy)

    # Set figure parameters
    ax.set(xlim=(0, L), ylim=(0, L))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Do each step, updating the quiver and plotting the new one
    for i in range(Nt):
        # print(i)
        x, y, vx, vy = step(x, y, vx, vy, theta, Rsq, x_obstacle, y_obstacle, eta, fov_angle, N, dt)
        q = update_quiver(q, x, y, vx, vy)
        clear_output(wait=True)
        display(fig)