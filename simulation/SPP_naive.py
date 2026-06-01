import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def update_boid_velocity_naive(
    self_position, self_velocity, positions, velocities,
    seperation_distance=20.0, alignment_distance=30.0, cohesion_distance=20.0,
    del_change=0.1, bounds=500.0
):
    # Constants
    PI = 3.14159
    PI_2 = PI * 2.0
    SPEED_LIMIT = 10.0
    UPPER_bounds = bounds
    LOWER_bounds = -UPPER_bounds

    # Compute zone radius and thresholds
    zone_radius = seperation_distance + alignment_distance + cohesion_distance
    separation_thresh = seperation_distance / zone_radius
    alignment_thresh = (seperation_distance + alignment_distance) / zone_radius
    zone_radius_squared = zone_radius * zone_radius

    # Initialize velocity
    velocity = np.array(self_velocity, dtype=float)
    limit = SPEED_LIMIT

    # Central attraction
    central = np.array([0.0, 0.0, 0.0])
    dir_vec = self_position - central
    dist = np.linalg.norm(dir_vec)
    if dist > 0:
        dir_vec[1] *= 2.5  # Amplify y-component
        dir_norm = dir_vec / dist
        velocity -= dir_norm * del_change * 6.0

    # Iterate over all boids
    for bird_position, bird_velocity in zip(positions, velocities):
        dir_vec = bird_position - self_position
        dist = np.linalg.norm(dir_vec)
        if dist < 0.0001:
            continue  # Skip self

        dist_squared = dist * dist
        if dist_squared > zone_radius_squared:
            continue  # Skip boids outside zone

        percent = dist_squared / zone_radius_squared

        if percent < separation_thresh:
            # Separation
            f = (separation_thresh / percent - 1.0) * del_change
            velocity -= (dir_vec / dist) * f if dist > 0 else np.zeros(3)
        elif percent < alignment_thresh:
            # Alignment
            threshold = alignment_thresh - separation_thresh
            adjusted_percent = (percent - separation_thresh) / threshold
            f = (0.5 - np.cos(adjusted_percent * PI_2) * 0.5 + 0.5) * del_change
            bird_velocity_norm = bird_velocity / np.linalg.norm(bird_velocity) if np.linalg.norm(bird_velocity) > 0 else np.zeros(3)
            velocity += bird_velocity_norm * f
        else:
            # Cohesion
            threshold = 1.0 - alignment_thresh
            adjusted_percent = (percent - alignment_thresh) / threshold
            f = (0.5 - (np.cos(adjusted_percent * PI_2) * -0.5 + 0.5)) * del_change
            velocity += (dir_vec / dist) * f if dist > 0 else np.zeros(3)

    
    # Limit speed
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude > limit:
        velocity = (velocity / velocity_magnitude) * limit

    # Add directional error
    error_direction = np.random.randn(3)
    error_direction /= np.linalg.norm(error_direction)
    error_magnitude = 0.05 * SPEED_LIMIT  # Adjust magnitude as needed
    velocity += error_direction * error_magnitude

    return velocity

def update_boid_velocity_vectorized(
    self_position, self_velocity, positions, velocities,
    seperation_distance=20.0, alignment_distance=30.0, cohesion_distance=20.0,
    del_change=0.1, bounds=500.0
):
    # Constants
    PI = 3.14159
    PI_2 = PI * 2.0
    SPEED_LIMIT = 10.0
    UPPER_bounds = bounds
    LOWER_bounds = -UPPER_bounds

    # Compute zone radius and thresholds
    zone_radius = seperation_distance + alignment_distance + cohesion_distance
    separation_thresh = seperation_distance / zone_radius
    alignment_thresh = (seperation_distance + alignment_distance) / zone_radius
    zone_radius_squared = zone_radius * zone_radius

    # Initialize velocity
    velocity = np.array(self_velocity, dtype=float)
    limit = SPEED_LIMIT

    # Central attraction
    central = np.array([0.0, 0.0, 0.0])
    dir_vec_central = self_position - central
    dist_central = np.linalg.norm(dir_vec_central)
    if dist_central > 0:
        dir_vec_central[1] *= 2.5  # Amplify y-component
        dir_norm_central = dir_vec_central / dist_central
        velocity -= dir_norm_central * del_change * 6.0

    # Compute interactions with all boids
    dir_vecs = positions - self_position  # Shape: (N, 3)
    dist_squared = np.sum(dir_vecs**2, axis=1)  # Shape: (N,)
    mask = (dist_squared > 0) & (dist_squared < zone_radius_squared)  # Exclude self and far boids

    if np.any(mask):
        dir_vecs_masked = dir_vecs[mask]
        dist_squared_masked = dist_squared[mask]
        distances_masked = np.sqrt(dist_squared_masked)
        percent = dist_squared_masked / zone_radius_squared

        # Separation
        separation_mask = percent < separation_thresh
        if np.any(separation_mask):
            f_separation = (separation_thresh / percent[separation_mask] - 1.0) * del_change
            dir_vec_norm_separation = dir_vecs_masked[separation_mask] / distances_masked[separation_mask][:, np.newaxis]
            separation_force = -np.sum(dir_vec_norm_separation * f_separation[:, np.newaxis], axis=0)
            velocity += separation_force

        # Alignment
        alignment_mask = (percent >= separation_thresh) & (percent < alignment_thresh)
        if np.any(alignment_mask):
            adjusted_percent = (percent[alignment_mask] - separation_thresh) / (alignment_thresh - separation_thresh)
            f_alignment = (0.5 - np.cos(adjusted_percent * PI_2) * 0.5 + 0.5) * del_change
            velocities_masked = velocities[mask]
            vel_norms = np.linalg.norm(velocities_masked[alignment_mask], axis=1)[:, np.newaxis]
            bird_velocity_norm = np.where(vel_norms > 0, velocities_masked[alignment_mask] / vel_norms, 0)
            alignment_adjustment = np.sum(bird_velocity_norm * f_alignment[:, np.newaxis], axis=0)
            velocity += alignment_adjustment

        # Cohesion
        cohesion_mask = percent >= alignment_thresh
        if np.any(cohesion_mask):
            threshold = 1.0 - alignment_thresh
            adjusted_percent = (percent[cohesion_mask] - alignment_thresh) / threshold
            f_cohesion = (0.5 - (np.cos(adjusted_percent * PI_2) * -0.5 + 0.5)) * del_change
            dir_vec_norm_cohesion = dir_vecs_masked[cohesion_mask] / distances_masked[cohesion_mask][:, np.newaxis]
            cohesion_adjustment = np.sum(dir_vec_norm_cohesion * f_cohesion[:, np.newaxis], axis=0)
            velocity += cohesion_adjustment

    # Limit speed
    velocity_magnitude = np.linalg.norm(velocity)
    if velocity_magnitude > limit:
        velocity = (velocity / velocity_magnitude) * limit

    # Add directional error
    error_direction = np.random.randn(3)
    error_direction /= np.linalg.norm(error_direction)
    error_magnitude = 0.05 * SPEED_LIMIT
    velocity += error_direction * error_magnitude

    return velocity

class SPPSimulation3D:
    def __init__(self, iterations=500, num_boids=500, bounds=500.0, del_change=0.033):
        """
        Initialize the 3D SPP simulation.
        
        Args:
            num_boids (int): Number of boids.
            bounds (float): Simulation boundary size (cube [-bounds, bounds]).
            del_change (float): Time interval between frames (seconds).
        """
        self.num_boids = num_boids
        self.bounds = bounds
        self.del_change = del_change
        self.iterations = iterations

        # Initialize positions and velocities in 3D using NumPy
        self.positions = np.random.rand(num_boids, 3) * 100 - 100
        self.velocities = (np.random.rand(num_boids, 3) - 0.5) * 10

        # Simulation parameters
        self.seperation_distance = 20.0
        self.alignment_distance = 30.0
        self.cohesion_distance = 20.0

        # Initialize lists to store positions and velocities over time
        self.position_history = np.zeros((iterations, num_boids, 3))
        self.velocity_history = np.zeros((iterations, num_boids, 3))

    def enforce_boundaries(self):
        """Keep boids within [-bounds, bounds] in all dimensions."""
        self.positions = np.clip(self.positions, -self.bounds, self.bounds)

    def update(self, frame):
        """Update boid positions and velocities, and update 3D plot."""
        print(f'Processing frame {frame}/{self.iterations}')
        # Create array for new velocities
        new_velocities = np.zeros_like(self.velocities)

        # Update velocities for each boid using the naive function
        for i in range(self.num_boids):
            new_velocities[i] = update_boid_velocity_vectorized(
                self_position=self.positions[i],
                self_velocity=self.velocities[i],
                positions=self.positions,
                velocities=self.velocities,
                seperation_distance=self.seperation_distance,
                alignment_distance=self.alignment_distance,
                cohesion_distance=self.cohesion_distance,
                del_change=self.del_change,
                bounds=self.bounds
            )

        # Update velocities
        self.velocities = new_velocities

        # Update positions: position += velocity * del_change * 15
        self.positions += self.velocities * self.del_change * 15.0

        # Record positions and velocities
        self.position_history[frame] = self.positions.copy()
        self.velocity_history[frame] = self.velocities.copy()

        # Enforce boundaries
        # self.enforce_boundaries()

        # Update 3D scatter plot
        self.scatter._offsets3d = (
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2]
        )
        return [self.scatter]

    def animate(self):
        """Set up and run the Matplotlib 3D animation."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-self.bounds, self.bounds)
        ax.set_ylim(-self.bounds, self.bounds)
        ax.set_zlim(-self.bounds, self.bounds)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D SPP Boid Simulation')
        self.scatter = ax.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
            s=10, c='blue'
        )

        ani = FuncAnimation(
            fig, self.update, frames=self.iterations, interval=1000 * self.del_change,
            blit=False,  # Blit=False for 3D scatter updates
            repeat=False # Stop animation after specified frames
        )
        plt.show()

        # After the animation is done, you can access the recorded data
        # self.position_history and self.velocity_history
        # Convert lists to NumPy arrays for easier handling
        self.position_history = np.array(self.position_history)
        self.velocity_history = np.array(self.velocity_history)

        np.savez('spp_simulation_data.npz', 
                 positions=self.position_history, 
                 velocities=self.velocity_history)

if __name__ == "__main__":
    # Initialize and run simulation
    sim = SPPSimulation3D(
        num_boids=1000,
        bounds=500.0,  # Smaller bounds for better 3D visualization
        del_change=0.033,  # ~30 FPS (1/30 seconds)
    )
    sim.animate()