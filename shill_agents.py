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
    
    # =========================================================================
    # Step 1: Computing the shill agent points
    # =========================================================================
    
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

    
    # =========================================================================
    # Step 2: Identifying two points on either side of the shill agents
    # =========================================================================
    
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


                # =============================================================
                # Step 3: Calculating the vector perpendicular to the line
                # =============================================================
                
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
            
                # =============================================================
                # Step 4: Ensuring perpendicular vector points outward
                # =============================================================
                
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

# Need to find out how to update the velocities based on this perp unit vector
