"""
# =============================================================================
# Module: model3.py
# =============================================================================

This module provides functions for the simulation of Model 3.

"""

# =============================================================================
# Importing Modules
# =============================================================================

# Import all intermediate and initialisation functions
from functions import *
from initialise import *
from params_default import params_default

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

# =============================================================================
# Model 3
# =============================================================================



# -----------------------------------------------------------------------------
# Find Neighbours
# -----------------------------------------------------------------------------

def proximity_lists(i, x, y, R_bird, R_min):
    """
    The function finds the neighbouring and too close birds for a specific bird
    """

    # Compute distances from bird "i" to all other birds
    distances = np.sqrt((x - x[i])**2 + (y - y[i])**2)
    
    # Define the set of neighbours that the bird can see
    neighbours = np.where(distances <= R_bird)[0]
        
    # Define the set of birds that are too close
    too_close = np.where(distances <= R_min)[0]

    # Excluding the bird itself
    neighbours = neighbours[neighbours != i]
    too_close = too_close[too_close != i]   
    
    return neighbours, too_close    

# -----------------------------------------------------------------------------
# Centre Velocity - Move towards the centre of the flock
# -----------------------------------------------------------------------------
  
def centre_vel(i, x, y, neighbours):
    
    # If there are no neighbors, no center-of-mass contribution
    if len(neighbours) == 0:
        return 0, 0

    # Compute center of mass of neighbors
    com_x = np.mean(x[neighbours])
    com_y = np.mean(y[neighbours])

    # Compute the vector pointing towards the center of mass
    direction_to_com = np.array([com_x - x[i][0], com_y - y[i][0]])

    # Normalize the vector
    normalized_direction = normalise(direction_to_com)

    # Extract the components
    centre_vx, centre_vy = normalized_direction

    return centre_vx, centre_vy
    
# -----------------------------------------------------------------------------
# Avoidance Velocity - avoid colissions with to align with other birds
# -----------------------------------------------------------------------------
    
def avoid_vel(i, x, y, too_close):
    
    # If there are no birds too close, no avoidance needed
    if len(too_close) == 0:
        return 0, 0

    # Initialize avoidance velocity components
    avoid_vx = 0
    avoid_vy = 0

    # Compute avoidance velocity contributions
    avoid_vx = (x[i][0] - x[too_close][0])
    avoid_vy = (y[i][0] - y[too_close][0])
    
    # Normalize the avoidance vector to ensure a unit direction
    normalised_avoidance = normalise(np.array([avoid_vx, avoid_vy]))

    # Extract the components
    avoid_vx, avoid_vy = normalised_avoidance

    return avoid_vx, avoid_vy

# -----------------------------------------------------------------------------
# Matching Velocity - match velocity to align with neighbours
# -----------------------------------------------------------------------------
    
def match_vel(i, vx, vy, neighbours):
    
    # Check if there are no neighbors
    if len(neighbours) == 0:
        return 0, 0

    # Calculate average neighbor velocity
    avg_vx = np.mean(vx[neighbours])
    avg_vy = np.mean(vy[neighbours])

    # Compute the velocity difference vector
    velocity_difference = np.array([avg_vx - vx[i][0], avg_vy - vy[i][0]])

    # Normalize the velocity difference to ensure a unit vector
    normalized_match = normalise(velocity_difference)

    # Extract the components
    match_vx, match_vy = normalized_match

    return match_vx, match_vy

# -----------------------------------------------------------------------------
# Migration Velocity - a common goal velocity for the flock
# -----------------------------------------------------------------------------

def migratory_vel(goal_x, goal_y):
    
    normalized_goal = normalise(np.array([goal_x, goal_y]))
    
    migrate_vx, migrate_vy = normalized_goal
    
    return migrate_vx, migrate_vy

# -----------------------------------------------------------------------------
# Obstacle Velocity - velocity to steer away from incoming obstacles
# -----------------------------------------------------------------------------

def obstacle_vel(i, x, y, vx, vy, R_min, R_obs, x_obstacle_list, y_obstacle_list, num_samples = 10):
    
    # -------------------------------------------------------------------------
    # Find obstacles which could result in a collision with bird's current traj
    # -------------------------------------------------------------------------
    
    # Initialise obstacle velocity as 0
    lam_o = 0
    obstacle_vx = 0
    obstacle_vy = 0
    
    # Get current location of the bird
    bird_loc = np.array([x[i][0], y[i][0]])  
    
    # Get current travelling vector of bird and normalise
    travel_vec = normalise(np.array([vx[i][0], vy[i][0]]))
    
    # Create uniformly spaced points along the bird's "line of travel"
    # This is maxed at the bird obstacle viewing radius
    lot_values = np.linspace(0, R_obs, num_samples).reshape(-1, 1) 
    lot_points = bird_loc + lot_values * travel_vec
    
    # Initialise a list to store the obstacle points that are too close to bird
    too_close_points = []
    lot_points_x = lot_points[:,0]
    lot_points_y = lot_points[:,1]
    
    # Loop over spaced points on the bird's line of travel
    for lot_x, lot_y in zip(lot_points_x, lot_points_y):
        
        # Iterate over each obstacle in the obstacle list
        for obstacle_idx, (x_obs, y_obs) in enumerate(zip(x_obstacle_list, y_obstacle_list)):
            
            # Calculate distance from line of travel point to the current obstacle's points
            distances = np.sqrt((np.array(x_obs) - lot_x)**2 + (np.array(y_obs) - lot_y)**2)
            
            # Find the obstacle(s) which is too close
            close_indices = np.where(distances <= R_min)[0]  
    
            # Store the specific point(s) on the obstacle which the bird could hit
            if len(close_indices) > 0:
                
                # Append the obstacle(s) index and point(s) index
                too_close_points.extend([(obstacle_idx, idx) for idx in close_indices])
    
    # Remove duplicates if necessary
    too_close_points = list(set(too_close_points))

    # If no obstacle points are too close then
    if not too_close_points:
        
        # The bird doesn't need a obstacle velocity component, return 0 
        return obstacle_vx, obstacle_vy, lam_o
    
    # -------------------------------------------------------------------------
    # If possible collision detected, find what object this is for
    # -------------------------------------------------------------------------
    
    # Start with a large value
    min_distance = np.inf  

    for obstacle_idx, point_idx in too_close_points:
        
        # Get the coordinates of the point
        point_x = x_obstacle_list[obstacle_idx][point_idx]
        point_y = y_obstacle_list[obstacle_idx][point_idx]
        
        # Calculate distance from bird's current position to the point
        distance_to_bird = np.sqrt((point_x - x[i][0])**2 + (point_y - y[i][0])**2)
        
        # Check if this point is closer than the current closest
        if distance_to_bird < min_distance:
            min_distance = distance_to_bird
            most_threatening_point = (obstacle_idx, point_idx)
            
    
    # Now "most_threatening_point" contains the obstacle index and point index
    obstacle_idx, point_idx = most_threatening_point
    
    # -------------------------------------------------------------------------
    # Find the weight for the obstacle velocity vector
    # -------------------------------------------------------------------------
    
    # scale min distance to be a value between 0 and 1 depending on the scale
    # of 0 to R_min
    p = 3  # Control steepness (tune this value)

    # Ensure min_distance is capped between 0 and R_min
    normalized_distance = min(min_distance / R_min, 1)
    
    # Calculate lam_o using polynomial scaling
    lam_o = (1 - normalized_distance) ** p

    # -------------------------------------------------------------------------
    # If possible collision detected, find nearest silhouette edge of object
    # -------------------------------------------------------------------------

    # Find line perpendicular to travelling vector of bird
    perp_vec = np.array([-travel_vec[1], travel_vec[0]])

    # Project each point of collision obstacle on to this perp line
    # Find points of object that can result in collision
    coll_obstacle_x = x_obstacle_list[obstacle_idx]
    coll_obstacle_y = y_obstacle_list[obstacle_idx]
    coll_obstacle_points = np.column_stack((coll_obstacle_x, coll_obstacle_y))
    
    # Initialize a list to store projections and corresponding points
    projections_with_points = []
    
    # Loop through each obstacle point
    for obs in coll_obstacle_points:
        # Calculate the projection of the point onto the perpendicular vector
        projection = np.dot(obs - bird_loc, perp_vec)
        
        # Store the projection along with the corresponding point
        projections_with_points.append((projection, obs))
    
    # Sort the projections to find the edges
    projections_with_points.sort(key=lambda x: x[0])  # Sort by projection value
    
    # Extract the silhouette edges
    left_edge_projection, left_edge_point = projections_with_points[0]  # Smallest projection
    right_edge_projection, right_edge_point = projections_with_points[-1]  # Largest projection
    
    silhouette_edges = [left_edge_point, right_edge_point]
    
    # Of these, find the one with minimum distance to the bird
    # This is the nearest silhouette edge of object
    closest_edge = min(silhouette_edges, key=lambda edge: abs(edge[0] - edge[1]))
    
    # The bird should steer to a point that is past the edge by a
    # distance at least the radius of the bird in order to clear the obstacle
    
    # Find the outwards perpendicular vector of the object and normalise
    # We do this by finding the two neighbouring points and computing a line
    # between them, and finding the vector perp to this line    
    
    # Exclude the closest_edge itself
    remaining_points = coll_obstacle_points[~(coll_obstacle_points == closest_edge).all(axis=1)]
    
    # Compute distances to closest_edge
    distances = np.linalg.norm(remaining_points - closest_edge, axis=1)
    
    # Find indices of the two closest points
    closest_indices = distances.argsort()[:2]  # Get indices of the two smallest distances
    
    # Get the two closest points
    side_points = remaining_points[closest_indices]

    # Calculate the line between them
    line_vec = side_points[0] - side_points[1]
    
    # Find the perpendicular vector and normalise
    obs_perp_vec = normalise(np.array([-line_vec[1], line_vec[0]]))
    
    # Find rough centre of obstacle
    centroid = np.mean(coll_obstacle_points, axis=0)
    
    # Make sure perp vector is pointing outwards from obstacle
    # Compute potential endpoints
    endpoint_1 = closest_edge + obs_perp_vec
    endpoint_2 = closest_edge - obs_perp_vec
    
    # Calculate distances from centroid
    distance_1 = np.linalg.norm(endpoint_1 - centroid)
    distance_2 = np.linalg.norm(endpoint_2 - centroid)
    
    # Choose the outward vector
    if distance_1 > distance_2:
        outward_perp_vec = obs_perp_vec
    else:
        outward_perp_vec = -obs_perp_vec
        
    # Multiply by radius of the bird to get target safe point
    safe_point = closest_edge + R_min * outward_perp_vec
    
    # Get velocity of bird to safe point and normalise
    steer_vec = normalise(safe_point - bird_loc) 
    obstacle_vx, obstacle_vy = steer_vec
         
    return obstacle_vx, obstacle_vy, lam_o


# -----------------------------------------------------------------------------
# Update velocities
# -----------------------------------------------------------------------------
       
def update_velocity(i, vx, vy, 
                    obstacle_vx, obstacle_vy, 
                    centre_vx, centre_vy, 
                    avoid_vx, avoid_vy, 
                    match_vx, match_vy, 
                    migrate_vx, migrate_vy, 
                    bird_speed_max, 
                    lam_a, lam_c, lam_m, lam_o, lam_g, lam_w,
                    vx_wind, vy_wind):
    
    # Update velocities with contributions
    vx_new = vx[i][0] + \
             lam_o * obstacle_vx + \
             lam_c * centre_vx + \
             lam_a * avoid_vx + \
             lam_m * match_vx + \
             lam_g * migrate_vx + \
             lam_w * vx_wind
             
    vy_new = vy[i][0] + \
             lam_o * obstacle_vy + \
             lam_c * centre_vy + \
             lam_a * avoid_vy + \
             lam_m * match_vy + \
             lam_g * migrate_vy + \
             lam_w * vy_wind

    # Limit speed to maximum
    current_speed = np.linalg.norm([vx_new, vy_new])
    if current_speed > bird_speed_max:
        scale = bird_speed_max / current_speed
        vx_new *= scale
        vy_new *= scale
    
    return vx_new, vy_new

# -----------------------------------------------------------------------------
# Update steps
# -----------------------------------------------------------------------------


def step(
    x,
    y,
    vx,
    vy,
    L,
    R_bird,
    R_min,
    N,
    dt,
    bird_speed_max,
    lam_a,
    lam_c,
    lam_m,
    lam_g,
    lam_o,
    lam_w,
    goal_x,
    goal_y,
    x_obstacle_list,
    y_obstacle_list,
    A_x,
    A_y,
    k,
    f,
    t
):
    
    '''
    Compute a step in the dynamics:
    - update the positions
    - compute the new velocities
    '''
    
    # Update positions based on velocity and time step
    x += vx * dt
    y += vy * dt
    
    # Apply periodic boundary conditions
    x %= L
    y %= L
    
    # Initialise the new velocities
    vx_new = np.zeros(N)
    vy_new = np.zeros(N)
    
    # Get combined wind velocity components
    vx_wind, vy_wind = wind_combined(x, y, t, A_x, A_y, k, f)
    
    # For each bird:
    for i in range(N):
        
        # Find neighbouring birds and those that are too close
        neighbours, too_close = proximity_lists(i, x, y, R_bird, R_min)
        
        # Obstacle avoidance component
        obstacle_vx, obstacle_vy = 0, 0
        
        # Center of mass component
        centre_vx, centre_vy = centre_vel(i, x, y, neighbours)
        
        # Bird avoidance component
        avoid_vx, avoid_vy = avoid_vel(i, x, y, too_close)
        
        # Matching component
        match_vx, match_vy = match_vel(i, vx, vy, neighbours)
        
        # Migrating component
        migrate_vx, migrate_vy = migratory_vel(goal_x, goal_y)
           
        # Update velocity with limits
        vx_new[i], vy_new[i] = update_velocity(i, vx, vy, 
                                    obstacle_vx, obstacle_vy, 
                                    centre_vx, centre_vy, 
                                    avoid_vx, avoid_vy, 
                                    match_vx, match_vy, 
                                    migrate_vx, migrate_vy, 
                                    bird_speed_max, 
                                    lam_a, lam_c, lam_m, lam_o, lam_g, lam_w,
                                    vx_wind[i], vy_wind[i])
    
    # Update new velocities
    vx = np.array(vx_new).reshape(-1, 1)
    vy = np.array(vy_new).reshape(-1, 1)
    
    return x, y, vx, vy

# -----------------------------------------------------------------------------
# Run Model 3
# -----------------------------------------------------------------------------

def run_model3(params, plot=False):

    # If no parameters are supplied,
    if params is None:

        # Then use default parameters
        params = params_default() 

    # Fetch the obstacles in the environment
    x_obstacle_list, y_obstacle_list, x_obstacle, y_obstacle = initialize_obstacles(
        L = params.L , 
        num_obstacles = params.num_obstacles, 
        nrows = params.nrows, 
        ncols = params.ncols, 
        shape = params.shape, 
        x_spacing = params.x_spacing,
        y_spacing = params.y_spacing,
        offset = params.offset, 
        beta = params.beta
    )

    # Fetch the initial birds in the environment
    x, y, vx, vy, _ = initialize_birds(
        N = params.N, 
        L = params.L, 
        v0 = params.v0, 
        theta_start = params.theta_start, 
        eta = params.eta,
        method = params.method
    )

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))

    # Initialize quiver plot
    q = ax.quiver(x, y, vx, vy)

    # Set figure parameters
    ax.set(xlim=(0, params.L), ylim=(0, params.L))
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Plot obstacle(s) - Plot the "list" to visualise the different obstaclces properly
    for xx, yy in zip(x_obstacle_list, y_obstacle_list):
        ax.plot(xx, yy, 'r-')

    # Wind visualization
    vx_wind, vy_wind = wind_combined(x, y, 0, params.A_x, params.A_y, params.k, params.f)
    wind_quiver = ax.quiver(0, 0, vx_wind.mean(), vy_wind.mean(), color='red', scale=1)

    # Metrics storage 
    dispersion_values = []
    offset_values = []
    clustering_coefficients = []

    # Function to update each frame
    for t in range(params.Nt):

        # Update bird positions and velocities
        x, y, vx, vy = step(
            x = x, 
            y = y, 
            vx = vx, 
            vy = vy, 
            L = params.L, 
            R_bird = params.R_bird, 
            R_min = params.r_min, 
            N = params.N, 
            dt = params.dt, 
            bird_speed_max = params.vmax,
            lam_a = params.lam_a, 
            lam_c = params.lam_c, 
            lam_m = params.lam_m, 
            lam_g = params.lam_g, 
            lam_o = params.lam_o,
            lam_w = params.lam_w,
            goal_x = params.goal_x, 
            goal_y = params.goal_y,
            x_obstacle_list = x_obstacle_list, 
            y_obstacle_list = y_obstacle_list,
            A_x = params.A_x,
            A_y = params.A_y,
            k = params.k,
            f = params.f,
            t = t
        )

        vx_wind, vy_wind = wind_combined(x, y, t * params.dt, params.A_x, params.A_y, params.k, params.f)

        # Append metric values
        dispersion_values.append(calculate_dispersion(x, y))
        offset_values.append(calculate_path_offset(x, y, params.goal_x, params.goal_y))
        clustering_coefficients.append(get_clustering_coefficient(vx, vy, params.v0, vx_wind, vy_wind, params.N))

        # Plot
        if plot:
            q = update_quiver(q, x, y, vx, vy)
            wind_quiver.set_UVC(vx_wind.mean(), vy_wind.mean())
            clear_output(wait=True)
            display(fig)