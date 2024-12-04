"""
# =============================================================================
# Module: model2.py
# =============================================================================

This module provides functions for the simulation of Model 2.

"""

# =============================================================================
# Importing Modules
# =============================================================================

# Import all intermediate functions
from .functions import *
from .model1 import *



def shill_agents(x_bird, y_bird, x_obs_in_radius, y_obs_in_radius, x_obstacle_list, y_obstacle_list):
    '''
    This function computes the closest "shill" agent(s) positions and their
    velocity vectors. It does so by:
        
    1. Computing the closest point(s) on the boundary of an obstacle to the 
       selected bird. This is a shill agent point.
       
    2. Identifying the two points on the obstacle that are on either side of
       a shill agent point.
    
    3. Calculating the unit vector perpendicular to the line formed by these 
       two points on either side of the shill agent.
    
    4. Ensuring the perpendicular vector points outward from the obstacle, 
       relative to the bird's position.
       
    Notes:
        The function works with knowledge that there could be multiple 
        positions which minimise the distance between bird and the obstacle,
        hence it finds multiple perpendicular vectors
    Inputs:
        x: Horizontal position of bird
        
        y: Vertical position of bird
        
        x_boundary: x-coords of obstacle points that are 
        within the radius of the bird
        
        y_boundary: y-coords of obstacle points that are 
        within the radius of the bird
    
    Outputs:
        x_d: Horizontal position of the shill agent
        
        y_d: Vertical position of the shill agent
    '''

    # -------------------------------------------------------------------------
    # Step 1: Computing the shill agent points
    # -------------------------------------------------------------------------

    # Compute distances from bird to each obstacle point within radius
    distances = np.sqrt((x_obs_in_radius - x_bird)**2 + (y_obs_in_radius - y_bird)**2)

    # Find the minimum distance(s)
    min_distance = np.min(distances)

    # Find all indices with the minimum distance
    closest_idxs = np.where(distances == min_distance)[0]  

    # Define position of shill agents as position of closest points
    x_shills = x_obs_in_radius[closest_idxs]
    y_shills = y_obs_in_radius[closest_idxs]  

    # Store all location of shill agents
    shill_locs = np.array([x_shills, y_shills])


    # -------------------------------------------------------------------------
    # Step 2: Identifying two points on either side of the shill agents
    # -------------------------------------------------------------------------

    # Storage list for perpendicular vectors
    perp_vecs = []

    # For each shill agent point, 
    for x_shill, y_shill in zip(shill_locs[0], shill_locs[1]):

        # Iterate over each obstacle,
        for x_obs, y_obs in zip(x_obstacle_list, y_obstacle_list):

            # Check if the shill agent belongs to this obstacle
            is_x_match = np.isclose(x_obs, x_shill) # (x_obs == x_shill)
            is_y_match = np.isclose(y_obs, y_shill) # (y_obs == y_shill)
            is_match = is_x_match & is_y_match  

            # If a match is found
            if np.any(is_match):  

                # Get the index of the shill agent
                shill_idx = np.where(is_match)[0][0]  

                # Find previous and next points (wrapping with modular)
                prev_idx = (shill_idx - 1) % len(x_obs)
                next_idx = (shill_idx + 1) % len(x_obs)

                # Get coordinates of previous and next points
                point_1 = (x_obs[prev_idx], y_obs[prev_idx])
                point_2 = (x_obs[next_idx], y_obs[next_idx])


                # -------------------------------------------------------------
                # Step 3: Calculating the vector perpendicular to the line
                # -------------------------------------------------------------

                # Unpack the coordinates of the two points
                x_1, y_1 = point_1
                x_2, y_2 = point_2

                # Compute the line vector
                line_x = x_2 - x_1
                line_y = y_2 - y_1

                # Compute the perpendicular vector
                perp_x = -line_y
                perp_y = line_x

                # Normalise the perpendicular vector so its a unit vector
                perp_x /= np.sqrt(perp_x**2 + perp_y**2)
                perp_y /= np.sqrt(perp_x**2 + perp_y**2)

                # -------------------------------------------------------------
                # Step 4: Ensuring perpendicular vector points outward
                # -------------------------------------------------------------

                # Carry out angle-based direction test to see if perp vector 
                # is facing in the correct direction

                # Compute direction vector from shill agent to bird
                dir_x = x_bird - x_shill
                dir_y = y_bird - y_shill

                # Normalise direction vector
                dir_x /= np.sqrt(dir_x**2 + dir_y**2)
                dir_y /= np.sqrt(dir_x**2 + dir_y**2)

                # Compute the dot product (cosine of the angle)
                dot_product = perp_x * dir_x + perp_y * dir_y

                # If the dot product is negative, the angle is greater than 
                # 90°, so flip
                if dot_product < 0:
                    perp_x = -perp_x
                    perp_y = -perp_y

                # Add perp vector to storage list
                perp_vecs.append((perp_x, perp_y))

    return shill_locs, perp_vecs



def update_theta(x, y, theta, Rsq, x_obstacle, y_obstacle, x_obstacle_list, y_obstacle_list, R_obs, eta, N, fov_angle):
    '''
    We will do this per bird, since we need to see if each one is within distance of obstacles or not
    '''
    # Initialize new theta array
    theta_new = theta.copy()
    
    # Get the mean theta from the neighbours
    mean_theta = get_mean_theta_neighbours(x, y, theta, Rsq, N)
    
    # Update theta based on obstacles
    for i in range(N):
        
        # Get obstacles within radius
        x_obs_in_radius, y_obs_in_radius, distances = get_obstacles_within_radius(
            x[i], y[i], theta[i], x_obstacle, y_obstacle, R_obs, fov_angle
        )
        
        # Check if there are obstacles within radius
        if len(distances) > 0:
            
            # Calculate shill agents and perpendicular vectors
            shill_locs, perp_vecs = shill_agents(
                x[i], y[i], x_obs_in_radius, y_obs_in_radius, x_obstacle_list, y_obstacle_list
            )
            
            # Compute distances between the bird and shill locations
            differences = shill_locs - np.array([x[i], y[i]])
            distances = np.linalg.norm(differences, axis=1)
            
            # Handle divide-by-zero in distances
            distances[distances == 0] = 1e-6  # Small epsilon value
            
            # Compute avoidance vectors normalized by distance
            avoidance_vectors = perp_vecs / distances[:, None]  # Ensure correct broadcasting
            
            # Sum up the avoidance vectors to get the net avoidance direction
            net_avoidance_vector = np.sum(avoidance_vectors, axis=0)
            
            # Calculate avoidance angle only if net_avoidance_vector is valid
            if np.linalg.norm(net_avoidance_vector) > 0:
                avoidance_theta = np.arctan2(net_avoidance_vector[1], net_avoidance_vector[0])
                
                # Calculate a weighted average between mean theta and avoidance theta
                avoidance_weight = 0.5  # Adjust as needed
                theta_new[i] = (1 - avoidance_weight) * mean_theta[i] + avoidance_weight * avoidance_theta
            
            else:
                # If avoidance vector is zero, fallback to mean theta
                theta_new[i] = mean_theta[i]
        
        else:
            # If no obstacles, use mean theta
            theta_new[i] = mean_theta[i]
    
    # Add noise to the updated theta values
    theta_new = add_noise_theta(theta_new, eta, N)
    
    return theta_new



def step(x, y, vx, vy, x_obstacle, y_obstacle, x_obstacle_list, y_obstacle_list, L, v0, theta, Rsq, R_obs,  eta, fov_angle, N, dt, v0_wind, v_wind_noise, wind_theta, wind_theta_noise):
    '''
    Compute a step in the dynamics:
    - update the positions
    - compute the new velocities
    '''
    x, y = update_positions(x, y, vx, vy, dt, L)
    theta = update_theta(x, y, theta, Rsq, x_obstacle, y_obstacle, x_obstacle_list, y_obstacle_list, R_obs, eta, N, fov_angle)
    
    vx_wind, vy_wind = wind_constant_with_noise(v0_wind, v_wind_noise, wind_theta, wind_theta_noise)
    
    #vx, vy = update_velocities(v0, theta, vx_wind, vy_wind, v_wind_max=10, alpha=0.5)
    vx, vy = update_velocities_original(v0, theta, vx_wind, vy_wind)
    
    return x, y, vx, vy, vx_wind, vy_wind



def run_model2(params, plot = False):

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
            x_obstacle_list = x_obstacle_list,
            y_obstacle_list = y_obstacle_list,
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