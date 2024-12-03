import numpy as np


# Starting Parameters
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


def initialize_birds_rectangle(N, L, v0, theta_start, eta):
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


# Obstacles
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
        # x_obs, y_obs = make_circular_obstacle(x_centres[i], y_centres[i], R=15, n=100)
        
        # Make rectangular obstacles
        x_obs, y_obs = make_rectangular_obstacle(x_centres[i], y_centres[i], L1=40, L2=4, n=100)
        
        x_obstacle_list.append(x_obs)
        y_obstacle_list.append(y_obs)
    
    # Concatenate lists for analysis
    x_obstacle = np.concatenate(x_obstacle_list)
    y_obstacle = np.concatenate(y_obstacle_list)
    
    return x_obstacle_list, y_obstacle_list, x_obstacle, y_obstacle


# WIND
def wind_constant_with_noise(v0_wind, v_wind_noise, wind_theta):
    '''
    Returns the x, y components of wind based on a constant angle.
    '''
    # Add random noise to the wind
    v0_wind += v_wind_noise * (np.random.rand(1) - 0.5)[0]

    # Get x, y velocity components
    vx_wind = v0_wind* np.cos(wind_theta)
    vy_wind = v0_wind* np.sin(wind_theta)
    
    return vx_wind, vy_wind

def wind_dynamic(t, v0_wind_base, v_wind_amplitude, v_wind_noise, wind_theta_base, wind_theta_variation):
    """
    Simulate wind velocity with a sinusoidal component and noise.
    Args:
        t: Current time step
        v0_wind_base: Base wind speed
        v_wind_amplitude: Amplitude of the sinusoidal variation
        v_wind_noise: Maximum random fluctuation in wind speed
        wind_theta_base: Base wind direction (in radians)
        wind_theta_variation: Maximum variation in wind direction
    """
    # Sinusoidal variation in wind speed
    v0_wind = v0_wind_base + v_wind_amplitude * np.sin(0.1 * t)
    
    # Add random noise to the wind speed
    v0_wind += v_wind_noise * (np.random.rand() - 0.5)
    
    # Gradually change wind direction
    wind_theta = wind_theta_base + wind_theta_variation * np.sin(0.05 * t)
    
    # Get x, y velocity components of wind
    vx_wind = v0_wind * np.cos(wind_theta)
    vy_wind = v0_wind * np.sin(wind_theta)
    
    return vx_wind, vy_wind

# Updating Steps
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

# Update Angle


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
                avoidance_weight = 0.5
                theta_new[i] = (1 - avoidance_weight) * mean_theta[i] + avoidance_weight * avoidance_theta
            
            else:
                # If that didn't work, just go to mean theta
                theta_new[i] = mean_theta[i]
    
        
        # If no obstacle, use theta from neighbours
        else:
            theta_new[i] = mean_theta[i]
        
    
    theta_new = add_noise_theta(theta_new, eta, N)
    
    return theta_new

# Update Velocity
def update_velocities(v0, theta, vx_wind, vy_wind):
    '''
    Update the velocities given theta, assuming a constant speed v0
    '''
    vx = v0 * np.cos(theta) + vx_wind
    vy = v0 * np.sin(theta) + vy_wind

    return vx, vy

# Update Quiver
def step(x, y, vx, vy, theta, Rsq, x_obstacle, y_obstacle, eta, fov_angle, N, dt, v0_wind, v_wind_noise, wind_theta, L, R_obs, v0):
    '''
    Compute a step in the dynamics:
    - update the positions
    - compute the new velocities
    '''
    x, y = update_positions(x, y, vx, vy, dt, L)
    theta = update_theta(x, y, theta, Rsq, x_obstacle, y_obstacle, R_obs, eta, N, fov_angle)
    
    vx_wind, vy_wind = wind_constant_with_noise(v0_wind, v_wind_noise, wind_theta)
    
    vx, vy = update_velocities(v0, theta, vx_wind, vy_wind)
    
    return x, y, vx, vy, vx_wind, vy_wind

def update_quiver(q,x,y,vx,vy):
    '''
    Update a quiver with new position and velocity information
    This is only used for plotting
    '''
    q.set_offsets(np.column_stack([x,y]))
    q.set_UVC(vx,vy)
    
    return q

# Clustering Coefficient
def get_clustering_coefficient(vx, vy, v0, vx_wind, vy_wind, N):
    
    # Sum the vx and vy components (with wind included)
    sum_terms_x = np.sum(vx)
    sum_terms_y = np.sum(vy)
    sum_terms = np.linalg.norm([sum_terms_x, sum_terms_y])
    
    # Get the expected v, v0 + v_wind
    # Expected total velocity magnitude per bird
    # v_expected = np.sqrt(v0**2 + vx_wind**2 + vy_wind**2) # doesn't work?
    
    # Average bird speed (calculate direc)
    v_expected = np.mean(np.sqrt(vx**2 + vy**2))
    
    # Calculate coefficient
    clustering_coefficient = (1/(N*v_expected))*sum_terms
    
    return clustering_coefficient


