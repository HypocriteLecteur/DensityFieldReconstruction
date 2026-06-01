import numpy as np
from scipy.stats import qmc

def generate_even_3d_points(N, min_x, max_x, min_y, max_y, min_z, max_z):
    """
    Generates N 3D points maximally spread (low discrepancy) within a cuboid.

    This function uses the Sobol sequence, a quasi-random method, which is superior 
    to standard uniform random sampling for space-filling design. The resulting 
    points are much more evenly distributed and less clustered.

    Args:
        N (int): The number of 3D points to generate.
        min_x (float): Minimum bound for the X-axis.
        max_x (float): Maximum bound for the X-axis.
        min_y (float): Minimum bound for the Y-axis.
        max_y (float): Maximum bound for the Y-axis.
        min_z (float): Minimum bound for the Z-axis.
        max_z (float): Maximum bound for the Z-axis.

    Returns:
        numpy.ndarray: A NumPy array of shape (N, 3) containing the [x, y, z] points.
    """
    if N <= 0:
        return np.array([])
    
    # 1. Define the lower and upper bounds for the 3 dimensions
    lower_bounds = np.array([min_x, min_y, min_z])
    upper_bounds = np.array([max_x, max_y, max_z])

    # 2. Initialize the Sobol sequence generator for 3 dimensions
    # Sobol is a type of low-discrepancy sequence designed to fill space evenly.
    sampler = qmc.Sobol(d=3, scramble=True)
    
    # 3. Generate N points in the unit hypercube [0, 1] x [0, 1] x [0, 1]
    # The 'scramble=True' argument helps improve the randomness quality for visualization.
    unit_points = sampler.random(n=N)

    # 4. Rescale the points to fit the desired cuboid bounds
    # qmc.scale applies the transformation: lower + (upper - lower) * unit_points
    points = qmc.scale(unit_points, lower_bounds, upper_bounds)
    
    return points

def generate_cluster_trajectories(
    dimensions: list,
    num_agents: int,
    num_clusters: int,
    time_steps: int,
    speed: float,
    noise_level: float,
    initial_cluster_radius: float = 0.1
) -> np.ndarray:
    """
    Generates trajectories for clustered agents in a 3D cuboid with boundary reflection.

    Agents move in clusters with aligned velocity vectors (constant speed). The velocity 
    direction is subject to individual agent-level random perturbation. The cluster's 
    shared base velocity is reflected when any agent hits a boundary, ensuring cohesion.

    Args:
        dimensions: A list of the cuboid boundaries [[xmin, xmax], [ymin, ymax], [zmin, zmax]].
        num_agents: Total number of agents to simulate.
        num_clusters: The number of distinct clusters.
        time_steps: The total number of time steps for the simulation.
        speed: The constant magnitude of velocity for all agents.
        noise_level: The standard deviation for the Gaussian noise applied to velocity direction.
        initial_cluster_radius: The base radius for initial agent placement (radii are randomized around this value).

    Returns:
        A numpy array of shape (num_agents, time_steps, 3) containing the trajectories.
    """
    # 1. Setup Environment and Initialization
    
    # Unpack dimensions
    dims = np.array(dimensions)
    min_coords = dims[:, 0]
    max_coords = dims[:, 1]
    
    # --- 1.1: Variable Cluster Assignment (Updated Logic for Agent/Radius Ratio) ---
    
    if num_agents < num_clusters:
        print("Error: Number of agents must be greater than or equal to number of clusters.")
        return np.zeros((num_agents, time_steps, 3))
        
    # 1. Generate random radii for each cluster (variable radius)
    # Radii are randomized between 50% and 150% of the input radius
    cluster_radii = initial_cluster_radius * np.random.uniform(0.8, 1.5, num_clusters)
    total_radius = np.sum(cluster_radii)

    # 2. Calculate agent counts proportional to their radius
    # Fractional count = Total Agents * (Radius_i / Total Radius)
    fractional_counts = num_agents * (cluster_radii / total_radius)
    
    # 3. Convert to integer counts, ensuring they sum exactly to num_agents
    agent_counts = np.floor(fractional_counts).astype(int)
    
    # Distribute the remainder agents (num_agents - sum(agent_counts)) one by one
    # to the clusters with the largest fractional parts (to ensure accurate rounding)
    remainder = num_agents - np.sum(agent_counts)
    
    if remainder > 0:
        # Calculate fractional parts
        fractional_parts = fractional_counts - agent_counts
        # Get indices of clusters with the largest fractional parts, descending
        sorted_indices = np.argsort(fractional_parts)[::-1]
        
        # Add the remainder agents to these clusters
        for i in range(remainder):
            agent_counts[sorted_indices[i]] += 1
            
    # Assign cluster IDs to agents
    cluster_ids = np.zeros(num_agents, dtype=int)
    start_index = 0
    for c, count in enumerate(agent_counts):
        cluster_ids[start_index:start_index + count] = c
        start_index += count
        
    # 2. Initialize Positions and Velocities
    
    # Initialize cluster centers (uniformly random within the bounds)
    cluster_centers = generate_even_3d_points(num_clusters, min_coords[0], max_coords[0], min_coords[1], max_coords[1], min_coords[2], max_coords[2])
    positions = np.zeros((num_agents, 3))
    
    # Initialize agent positions around their cluster center (using variable radii)
    start_index = 0
    for c, count in enumerate(agent_counts):
        indices = np.arange(start_index, start_index + count)
        num_c_agents = len(indices)
        current_radius = cluster_radii[c]
        
        if num_c_agents == 0:
            continue

        # Generate random displacement vectors in a sphere of radius 'current_radius'
        displacements = np.zeros((num_c_agents, 3))
        count_d = 0
        while count_d < num_c_agents:
            # Generate points in a cube [-1, 1]
            r = np.random.uniform(-1, 1, 3)
            # Check if point is in unit sphere
            if np.linalg.norm(r) < 1:
                # Scale by radius
                displacements[count_d, :] = r * current_radius
                count_d += 1
        
        # Set agent positions
        positions[indices] = cluster_centers[c] + displacements
        
        # Ensure initial positions are strictly within bounds
        positions[indices] = np.clip(
            positions[indices], 
            min_coords + 1e-6, 
            max_coords - 1e-6
        )
        start_index += count

    # Initialize cluster velocities (random direction, normalized to speed)
    cluster_velocities = np.random.randn(num_clusters, 3)
    # Normalize and scale by speed (This represents the *core* direction, not agent speed)
    norms = np.linalg.norm(cluster_velocities, axis=1, keepdims=True)
    cluster_velocities = (cluster_velocities / norms) * speed
    
    # Initialize agent trajectories array
    trajectories = np.zeros((num_agents, time_steps, 3))
    trajectories[:, 0, :] = positions.copy()
    
    # Pre-allocate array for noise to avoid reallocating inside the loop
    agent_noise_buffer = np.zeros((num_agents, 3)) 
    
    # 3. Simulation Loop
    
    for t in range(1, time_steps):
        
        # Step 3.1: Get Base Cluster Velocity
        # This is the shared direction vector for all agents in the cluster.
        agent_base_velocities = cluster_velocities[cluster_ids]
        
        # Step 3.2: Agent-Level Perturbation (Individual Noise)
        
        # Generate Gaussian noise for each agent
        agent_noise = np.random.normal(0, noise_level, size=(num_agents, 3))
        
        # Apply noise to the base velocity
        noisy_velocities = agent_base_velocities + agent_noise
        
        # Re-normalize to maintain constant speed 'speed' for each agent
        norms = np.linalg.norm(noisy_velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Protect against zero norm
        current_agent_velocities = (noisy_velocities / norms) * speed 
        
        # Step 3.3: Tentative Position Update
        new_positions = positions + current_agent_velocities
        
        # Step 3.4: Boundary Reflection (Cohesive Cluster Logic)
        
        # Check for boundary violations using the tentative positions
        hit_min = new_positions < min_coords
        hit_max = new_positions > max_coords
        
        # Determine which clusters need reflection
        for c in range(num_clusters):
            indices = np.where(cluster_ids == c)[0]
            
            # Check if ANY agent in the cluster hit a boundary on any axis
            cluster_violation = np.any(hit_min[indices] | hit_max[indices], axis=0)
            
            # If a violation occurred, reflect the cluster's shared base velocity
            for i in range(3): # Iterate over x, y, z axes
                if cluster_violation[i]:
                    cluster_velocities[c, i] *= -1
                    
        # Step 3.5: Final Position Calculation and Correction
        
        # If a cluster reflected in 3.4, its base velocity has changed.
        # We must recalculate the final agent velocity for this step using the 
        # reflected cluster velocity plus the *same* agent noise from 3.2.
        
        # Recalculate base velocity using the (potentially reflected) cluster velocity
        agent_base_velocities_reflected = cluster_velocities[cluster_ids]
        
        # Recalculate noisy velocity using the reflected base + agent noise
        noisy_velocities_reflected = agent_base_velocities_reflected + agent_noise
        
        # Re-normalize again to maintain constant speed
        norms = np.linalg.norm(noisy_velocities_reflected, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 
        final_agent_velocities = (noisy_velocities_reflected / norms) * speed
        
        # Final position update
        new_positions = positions + final_agent_velocities
        
        # Final boundary correction (Clip to ensure agents stay within boundaries)
        for i in range(3):
            new_positions[:, i] = np.clip(new_positions[:, i], min_coords[i], max_coords[i])

        # Step 3.6: Finalize Step
        positions = new_positions.copy()
        trajectories[:, t, :] = positions
        
    return trajectories