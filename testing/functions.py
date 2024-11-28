# =============================================================================
# Importing Packages
# =============================================================================  

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# =============================================================================
# Obstacle Functions
# =============================================================================

def make_circular_obstacle(x_centre, y_centre, R, n=20):
    '''
    Returns x,y points defining a circular obstacle
    '''
    angles = np.linspace(0, 2 * np.pi, n)
    
    x = x_centre + R*np.cos(angles)
    y = y_centre + R*np.sin(angles)
    
    return x, y

def make_rectangular_obstacle(x_centre, y_centre, L1, L2, n=25):
    '''
    Returns x,y points defining a rectangular obstacle
    '''
    
    # Number of points per side
    points_per_side = n // 4
    
    # Half lengths for width and height
    l1 = L1 / 2
    l2 = L2 / 2
    
    # Corners of the rectangle
    top_left = [x_centre - l1, y_centre + l2]
    top_right = [x_centre + l1, y_centre + l2]
    bottom_left = [x_centre - l1, y_centre - l2]
    bottom_right = [x_centre + l1, y_centre - l2]
    
    # Initialize lists for x and y points
    x_points = []
    y_points = []
    
    # Generate points along each side
    # Top edge (left to right)
    x_points.extend(np.linspace(top_left[0], top_right[0], points_per_side))
    y_points.extend([top_left[1]] * points_per_side)
    
    # Right edge (top to bottom)
    x_points.extend([top_right[0]] * points_per_side)
    y_points.extend(np.linspace(top_right[1], bottom_right[1], points_per_side))
    
    # Bottom edge (right to left)
    x_points.extend(np.linspace(bottom_right[0], bottom_left[0], points_per_side))
    y_points.extend([bottom_left[1]] * points_per_side)
    
    # Left edge (bottom to top)
    x_points.extend([bottom_left[0]] * points_per_side)
    y_points.extend(np.linspace(bottom_left[1], top_left[1], points_per_side))
    
    return x_points, y_points

def get_obstacle_centre_grid(L, num_obstacles, nrows, ncols):
    '''
    Define the centre of obstacles based on a grid.
    '''
    x_spacing = L / (ncols + 1)
    y_spacing = L / (nrows + 1)
    
    x_centres = []
    y_centres = []

    # Calc grid positions
    cnt = 0
    for i in range(nrows):
        for j in range(ncols):
            if cnt > num_obstacles:
                break
        
            # Calculate centre positions
            x_centre = (j + 1) * x_spacing
            y_centre = (i + 1) * y_spacing
            x_centres.append(x_centre)
            y_centres.append(y_centre)
            cnt += 1
    
    return x_centres, y_centres

def get_obstacles(L, num_obstacles, nrows, ncols):
    '''
    Call the obstacle functions and get lists of their x, y points
    '''
    
    x_centres, y_centres = get_obstacle_centre_grid(L, num_obstacles, nrows=nrows, ncols=ncols)
    
    # Initalise lists
    x_obstacle_list = []
    y_obstacle_list = []
    
    for i in range(num_obstacles):
        # Make circular obstacles
        x_obs, y_obs = make_circular_obstacle(x_centres[i], y_centres[i], R=0.25, n=20)
        
        # Make rectangular obstacles
        # x_obs, y_obs = make_rectangular_obstacle(x_centres[i], y_centres[i], L1=1, L2=0.2, n=25)
        
        x_obstacle_list.append(x_obs)
        y_obstacle_list.append(y_obs)
    
    # Concatenate lists for analysis
    x_obstacle = np.concatenate(x_obstacle_list)
    y_obstacle = np.concatenate(y_obstacle_list)
    
    return x_obstacle_list, y_obstacle_list, x_obstacle, y_obstacle


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
    
    # Filter by field of view
    is_in_fov = angle_diff <= fov_angle
    x_obs_in_radius = x_obs_in_radius[is_in_fov]
    y_obs_in_radius = y_obs_in_radius[is_in_fov]
    distances = distances[is_in_fov]
    
    return x_obs_in_radius, y_obs_in_radius, distances
  
# =============================================================================
# Bird Functions
# =============================================================================


def add_noise_theta(theta, eta, N):
    '''
    Update theta with a random amount of noise between -eta/2 and eta/2
    '''
    theta += eta * (np.random.rand(N, 1) - 0.5)
    
    return theta

def initialize_birds_random(N, L, v0):
    '''
    Set initial positions, direction, and velocities 
    '''
    # Bird positions
    x = np.random.rand(N, 1)*L
    y = np.random.rand(N, 1)*L

    # Bird velocities
    theta = 2 * np.pi * np.random.rand(N, 1)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)

    return x, y, vx, vy, theta

def initialize_birds_uniform(N, L, v0, theta_start, eta):
    '''
    Set initial positions as a uniform placement starting at the edge of the box.
    Set direction and velocity to be uniform with a small amount of noise 
    '''
    
    # Set bird initial flock as a square
    N_per_side = int(np.sqrt(N))
    
    # Midpoint of area 
    midpoint = L//2
    
    # Define the x locations (start in centre of perimeter)
    half_length = N_per_side // 2
    start = midpoint - half_length
    x_locs = np.arange(start, start + N_per_side)
    
    # Define the y locations (start from bottom)
    y_locs = np.arange(0, N_per_side)
    
    # Define bird starting points
    # Initialise lists
    x = []
    y = []
    
    for x_loc in x_locs:
        for y_loc in y_locs:
            x.append([x_loc])
            y.append([y_loc])
    
    # Turn into numpy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Bird Angle
    # Make all birds same starting angle
    theta = np.ones((len(x),1))*theta_start
    
    # Add noise
    theta = add_noise_theta(theta, eta, N)
    
    # Bird velocities
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
  
    return x, y, vx, vy, theta

def proximity_lists(i, x, y, R, r_min):
    """
    The function finds the neighbouring and too close birds for a specific bird
    """

    # Compute distances from bird "i" to all other birds
    distances = np.sqrt((x - x[i])**2 + (y - y[i])**2)
    
    # Define the set of neighbours that the bird can see
    neighbours = np.where(distances < R)[0]
        
    # Define the set of birds that are too close
    too_close = np.where(distances < r_min)[0]

    # Excluding the bird itself
    neighbours = neighbours[neighbours != i]
    too_close = too_close[too_close != i]   
    
    return neighbours, too_close    

# -----------------------------------------------------------------------------
# Centre Velocity - Move towards the centre of the flock
# -----------------------------------------------------------------------------
  
def centre_vel(i, x, y, neighbours):
    
    if len(neighbours) == 0:
        # If there are no neighbors, no center-of-mass contribution
        return 0, 0

    # Compute center of mass of neighbors
    com_x = np.mean(x[neighbours])
    com_y = np.mean(y[neighbours])

    # Compute velocity components towards center of mass
    centre_vx = (com_x - x[i])
    centre_vy = (com_y - y[i])

    return centre_vx, centre_vy
    
# -----------------------------------------------------------------------------
# Avoidance Velocity - avoid colissions with to align with other birds
# -----------------------------------------------------------------------------
    
def avoid_vel(i, x, y, too_close):
    """
    Determine the velocity away from potential bird-bird collisions
    """

    # Initialize avoidance velocity components
    avoid_vx = 0
    avoid_vy = 0

    # Loop through each bird in the too_close set
    for j in too_close:
        # Compute avoidance velocity contributions
        avoid_vx += (x[i] - x[j])
        avoid_vy += (y[i] - y[j])

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
    
    # Compute velocity matching component
    match_vx = (avg_vx - vx[i])
    match_vy = (avg_vy - vy[i])
    
    return match_vx, match_vy

# -----------------------------------------------------------------------------
# Obstacle Velocity - velocity to avoid collision with obstacles
# -----------------------------------------------------------------------------

def obstacle_vel(i, x, y, vx, vy, R_obs, num_samples, x_obstacle_list, y_obstacle_list):

    # Initialise velocities
    obstacle_vx = 0
    obstacle_vy = 0
    
    # Find current travelling vector of bird and normalize
    forward_vector_x = vx[i] / np.linalg.norm([vx[i], vy[i]])
    forward_vector_y = vy[i] / np.linalg.norm([vx[i], vy[i]])

    # Create uniformly spaced points along the bird's line of sight vector
    t_values = np.linspace(0, 1, num_samples)
    sampled_points_x = x[i] + t_values * forward_vector_x * R_obs
    sampled_points_y = y[i] + t_values * forward_vector_y * R_obs
    
    # Initialize a list to store the obstacle points that are too close
    too_close_points = []
    
    # Loop over sampled points on the bird's line of sight
    for sample_x, sample_y in zip(sampled_points_x, sampled_points_y):
        
        # Iterate over each obstacle in the obstacle list
        for obstacle_idx, (x_obs, y_obs) in enumerate(zip(x_obstacle_list, y_obstacle_list)):
            
            # Calculate distances from the sampled point to the current obstacle's points
            distances = np.sqrt((np.array(x_obs) - sample_x)**2 + (np.array(y_obs) - sample_y)**2)
            
            # Find indices of obstacle points within r_min
            close_indices = np.where(distances <= r_min)[0]  
            # Indices within the current obstacle's points
    
            # If there are any points too close, store them
            if len(close_indices) > 0:
                
                # Append the obstacle index and point indices
                too_close_points.extend([(obstacle_idx, idx) for idx in close_indices])
    
    # Remove duplicates if necessary
    too_close_points = list(set(too_close_points))

    # If no obstacle points are too close then
    if not too_close_points:
        
        # Return the 0 state of the obstacle velocity
        return obstacle_vx, obstacle_vy

    # Else if there are points which are too close, then find most threatening
    min_distance = np.inf  # Start with a large value

    for obstacle_idx, point_idx in too_close_points:
        
        # Get the coordinates of the point
        point_x = x_obstacle_list[obstacle_idx][point_idx]
        point_y = y_obstacle_list[obstacle_idx][point_idx]
        
        # Calculate distance from bird's current position to the point
        distance_to_bird = np.sqrt((point_x - x[i])**2 + (point_y - y[i])**2)
        
        # Check if this point is closer than the current closest
        if distance_to_bird < min_distance:
            min_distance = distance_to_bird
            most_threatening_point = (obstacle_idx, point_idx)
    
    # Now `most_threatening_point` contains the obstacle index and point index
    obstacle_idx, point_idx = most_threatening_point
    threatening_point_x = x_obstacle_list[obstacle_idx][point_idx]
    threatening_point_y = y_obstacle_list[obstacle_idx][point_idx]
    
    # Calculate the vector from bird to the most threatening point
    threatening_vector_x = threatening_point_x - x[i]
    threatening_vector_y = threatening_point_y - y[i]
    
    # Normalize this
    distance_to_threat = np.sqrt(threatening_vector_x**2 + threatening_vector_y**2)
    threatening_vector_x /= distance_to_threat
    threatening_vector_y /= distance_to_threat
    
    # Compute the cross product to decide steering direction
    # The sign of the cross product indicates the orientation of the vectors relative to each other.
    cross_product = threatening_vector_x * forward_vector_y - threatening_vector_y * forward_vector_x
                
    # If the bird is heading nearly directly at the obstacle 
    # the two vectors are ~parallel, the cross product will be ~zero,
    if abs(cross_product) < 0.1 :  
        
        # Compute a steering vector away as perp vector of travelling vector
        steer_vx = forward_vector_y
        steer_vy = forward_vector_x
    
    else:
        
        # Compute a steering vector away as opposite direction of threat vector
        steer_vx = -threatening_vector_x
        steer_vy = -threatening_vector_y
        
           
    # then make coeff depend on min distance to get magnistude of threat vector
    # Normalized between 0,1
    # Coeff is a value between 1 and 10
    coeff =  1 / (min_distance / r_min)
    

    # Compute obstacle avoidance vector 
    obstacle_vx = coeff * steer_vx 
    obstacle_vy = coeff * steer_vy
         
    return obstacle_vx, obstacle_vy
         
         
def update_velocity(i, vx, vy, \
                    obstacle_vx, obstacle_vy, \
                    centre_vx, centre_vy, \
                    avoid_vx, avoid_vy, \
                    match_vx, match_vy, \
                    vmax,
                    lam_a,
                    lam_c,
                    lam_m,
                    lam_o):
    
    # Update velocities with contributions
    vx_new = lam_o * obstacle_vx + \
             lam_c * centre_vx + \
             lam_a * avoid_vx + \
             lam_m * match_vx
             
    vy_new = lam_o * obstacle_vy + \
             lam_c * centre_vy + \
             lam_a * avoid_vy + \
             lam_m * match_vy

    # Compute current speeds
    speed = np.sqrt(vx[i]**2 + vy[i]**2)
    scale_factor = min(1, vmax / speed)
    
    # Apply scaling to limit speed
    vx_new *= scale_factor
    vy_new *= scale_factor
    
    return vx_new, vy_new

    
