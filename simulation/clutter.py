import torch
import math
import numpy as np # Keep numpy for printing and initial non-tensor operations
import time
import torch.nn.functional as f

# --- JIT-COMPILED KERNEL FOR REPULSION CALCULATION ---
# This function eliminates Python loop overhead for the most critical part of the simulation.
@torch.jit.script
def _jit_cell_interaction_kernel(P: torch.Tensor, cell_start: torch.Tensor, n_cells: torch.Tensor, 
                                 repulsion_R: float, repulsion_K: float, 
                                 offsets: torch.Tensor, total_cells: int, 
                                 n_cells_x: int, n_cells_xy: int) -> torch.Tensor:
    """
    Calculates net repulsion forces for all agents using the cell list.
    Compiled with torch.jit.script to minimize Python overhead.
    """
    F_rep = torch.zeros_like(P)
    repulsion_R_sq = repulsion_R * repulsion_R
    
    for cell_1d_idx in range(total_cells):
        start_i = cell_start[cell_1d_idx]
        end_i = cell_start[cell_1d_idx + 1]
        if start_i == end_i:
            continue
            
        P_i = P[start_i:end_i]
        N_i = P_i.shape[0]

        # 1D index to 3D coordinates (JIT-compatible)
        c_z = cell_1d_idx // n_cells_xy
        rem_z = cell_1d_idx % n_cells_xy
        c_y = rem_z // n_cells_x
        c_x = rem_z % n_cells_x
        
        cell_coords_i = torch.tensor([c_x, c_y, c_z], dtype=torch.long, device=P.device)
        
        # Iterate through the 27 neighbor cells (j)
        for offset in offsets:
            cell_coords_j = cell_coords_i + offset
            
            # Check if the neighbor cell is within the simulation boundaries
            if not torch.all((cell_coords_j >= 0) & (cell_coords_j < n_cells)):
                continue
            
            # Convert back to 1D index
            cell_j_idx = cell_coords_j[0] + \
                         cell_coords_j[1] * n_cells_x + \
                         cell_coords_j[2] * n_cells_xy
            
            start_j = cell_start[cell_j_idx]
            end_j = cell_start[cell_j_idx + 1]
            if start_j == end_j:
                continue

            P_j = P[start_j:end_j]
            
            # --- Vectorized Interaction Calculation (runs on GPU/Vectorized CPU) ---
            delta_P = P_i[:, None, :] - P_j[None, :, :] # N_i x N_j x 3
            r_sq = torch.sum(delta_P**2, dim=2)        # N_i x N_j
            
            mask = (r_sq < repulsion_R_sq)
            
            # Self-interaction check: must zero out when i and j refer to the same group/cell
            if cell_1d_idx == cell_j_idx:
                # Mask out diagonal (self-interaction, r_sq=0) in a JIT-compatible way
                mask = mask & (~torch.eye(N_i, device=P.device, dtype=torch.bool))
            
            if not torch.any(mask):
                continue

            r_sq_masked = r_sq * mask
            # Repulsion force calculation (Inverse Cube: K * delta_P / r^3)
            # Add epsilon for stability before taking the power
            r_cube = (r_sq_masked + 1e-9).pow(1.5) 
            
            # F = K * (P_i - P_j) / r^3
            F_interaction = repulsion_K * (delta_P / r_cube.unsqueeze(-1))
            
            F_interaction_masked = F_interaction * mask.unsqueeze(-1)
            
            # Sum forces on agents in cell i from neighbors in cell j
            F_rep_i = torch.sum(F_interaction_masked, dim=1)
            
            # Add to total force array
            F_rep[start_i:end_i] += F_rep_i
            
    return F_rep

class CuboidSimulation:
    """
    Simulates the movement of agents within a 3D cuboid using PyTorch and GPU
    acceleration. It utilizes the Cell List method for O(N) neighbor finding
    to ensure scalability for thousands of agents and avoid MemoryError.
    """
    def __init__(self, x_bounds, y_bounds, z_bounds, density, constant_speed,
                 noise_strength, repulsion_K, repulsion_R):
        """
        Initializes the simulation environment and agents using PyTorch tensors.

        Args:
            x_bounds (tuple): (xmin, xmax) for the cuboid.
            y_bounds (tuple): (ymin, ymax) for the cuboid.
            z_bounds (tuple): (zmin, zmax) for the cuboid.
            density (float): Number of agents per unit volume (agents/m^3).
            constant_speed (float): The initial constant speed magnitude for all agents.
            noise_strength (float): The magnitude of random acceleration perturbation.
            repulsion_K (float): The strength constant for the short-range repulsion force.
            repulsion_R (float): The effective short-range radius for repulsion.
        """
        # --- Device and Bounds Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Store bounds and parameters (as tensors for device)
        self.bounds = torch.tensor([x_bounds, y_bounds, z_bounds], dtype=torch.float32, device=self.device)
        self.box_size = self.bounds[:, 1] - self.bounds[:, 0]
        
        self.noise_strength = noise_strength
        self.repulsion_K = repulsion_K
        self.repulsion_R = repulsion_R
        self.constant_speed = constant_speed

        # Calculate number of agents
        volume = self.box_size[0].item() * self.box_size[1].item() * self.box_size[2].item()
        self.N = max(1, math.ceil(density * volume))
        print(f"Initializing {self.N} agents in a volume of {volume:.2f} cubic units.")
        
        # --- Cell List Setup for O(N) Neighbor Finding ---
        # Cell size must be >= repulsion_R
        self.cell_size = self.repulsion_R * 1.01 
        self.n_cells = torch.floor(self.box_size / self.cell_size).long()
        self.total_cells = torch.prod(self.n_cells).item()
        print(f"Dividing space into {self.n_cells.tolist()} grid cells ({self.total_cells} total).")

        # Pre-calculate constants needed for JIT kernel
        self.n_cells_x = self.n_cells[0].item()
        self.n_cells_xy = (self.n_cells[0] * self.n_cells[1]).item()
        self.offsets = torch.tensor([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]], 
                                     dtype=torch.long, device=self.device)

        # Initialize positions (P) and velocities (V) on device
        self._initialize_agents()
        self.cell_indices = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.cell_start = torch.zeros(self.total_cells + 1, dtype=torch.long, device=self.device)
        self.cell_list = torch.zeros(self.N, dtype=torch.long, device=self.device)
        
        # Timing storage
        self.total_timings = {'cell_list': 0.0, 'repulsion': 0.0, 'other': 0.0}
        
        self._build_cell_list()

    def _initialize_agents(self):
        """
        Initializes agent positions uniformly and velocities with random,
        isotropic directions but constant speed, using PyTorch tensors on the device.
        """
        # 1. Initialize Positions (P): N x 3 tensor
        self.P = torch.zeros((self.N, 3), dtype=torch.float32, device=self.device)
        for i in range(3):
            # Uniformly distribute across the cuboid volume
            self.P[:, i] = self.bounds[i, 0] + (self.bounds[i, 1] - self.bounds[i, 0]) * torch.rand(self.N, device=self.device)

        # 2. Initialize Velocities (V): N x 3 tensor
        V_random = torch.rand((self.N, 3), device=self.device) * 2 - 1

        # Normalize to unit vectors and apply constant speed
        V_norms = torch.linalg.norm(V_random, dim=1, keepdim=True)
        V_directions = V_random / V_norms
        self.V = V_directions * self.constant_speed

    def _build_cell_list(self):
        """
        Calculates which cell each agent belongs to and sorts the agents,
        updating cell_indices, cell_start, and cell_list.
        O(N log N) dominated by sorting, but much faster than O(N^2) interaction.
        """
        # 1. Calculate cell coordinates (i, j, k) for each agent
        cell_coords = torch.floor((self.P - self.bounds[:, 0]) / self.cell_size).long()
        
        # Clamp cell coordinates to prevent out-of-bounds index errors
        for i in range(3):
            cell_coords[:, i] = torch.clamp(cell_coords[:, i], 0, self.n_cells[i] - 1)

        # 2. Map 3D coordinates to a 1D cell index
        # index = i + j*N_x + k*(N_x*N_y)
        cell_idx = cell_coords[:, 0] + \
                   cell_coords[:, 1] * self.n_cells[0] + \
                   cell_coords[:, 2] * (self.n_cells[0] * self.n_cells[1])
        
        # 3. Sort agents based on cell index
        sorted_indices = torch.argsort(cell_idx)
        self.P = self.P[sorted_indices]
        self.V = self.V[sorted_indices]
        self.cell_indices = cell_idx[sorted_indices]
        
        # 4. Determine start index for each cell
        self.cell_start.fill_(0)
        # Count the number of agents in each cell
        counts = torch.bincount(self.cell_indices, minlength=self.total_cells)
        # Compute the cumulative sum to get the starting index of agents for each cell
        self.cell_start[1:] = torch.cumsum(counts, dim=0)

    def _calculate_repulsion_cell_list(self):
        """
        Calls the JIT-compiled kernel to efficiently calculate the net short-range 
        repulsion force using the Cell List method.
        """
        
        F_rep = _jit_cell_interaction_kernel(
            self.P, 
            self.cell_start, 
            self.n_cells, 
            self.repulsion_R, 
            self.repulsion_K, 
            self.offsets, 
            self.total_cells,
            self.n_cells_x,
            self.n_cells_xy
        )
                
        return F_rep

    def step(self, dt):
        """
        Performs one simulation step (velocity update, boundary check, position update).
        Uses PyTorch for GPU acceleration and includes timing for major phases.
        
        Args:
            dt (float): Time step size.

        Returns:
            tuple: (numpy.ndarray, dict) - The new positions (P) and a dictionary of step timings.
        """
        # --- Timing Setup ---
        is_cuda = self.device.type == 'cuda'
        
        if is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t_start = time.perf_counter()
        
        current_step_timings = {}

        # 1. Update Cell List (O(N log N))
        # self._build_cell_list() 
        
        if is_cuda:
            t_mid_1 = torch.cuda.Event(enable_timing=True)
            t_mid_1.record()
            torch.cuda.synchronize()
            current_step_timings['cell_list'] = start_event.elapsed_time(t_mid_1) / 1000.0 # seconds
            t_start = t_mid_1
        else:
            t_mid_1 = time.perf_counter()
            current_step_timings['cell_list'] = t_mid_1 - t_start
            t_start = t_mid_1
        
        # 2. Calculate Forces (O(N) due to Cell List)
        # F_rep = self._calculate_repulsion_cell_list()

        if is_cuda:
            t_mid_2 = torch.cuda.Event(enable_timing=True)
            t_mid_2.record()
            torch.cuda.synchronize()
            current_step_timings['repulsion'] = t_start.elapsed_time(t_mid_2) / 1000.0 # seconds
            t_start = t_mid_2
        else:
            t_mid_2 = time.perf_counter()
            current_step_timings['repulsion'] = t_mid_2 - t_start
            t_start = t_mid_2

        # 3. Add Random Perturbation (Acceleration)
        # N x 3 array of random accelerations on the device
        A_rand = torch.rand((self.N, 3), device=self.device) * 2 * self.noise_strength - self.noise_strength

        # 4. Total Acceleration (assuming mass=1)
        # A_total = A_rand + F_rep 
        A_total = A_rand # No Repulsion

        # 5. Velocity Update (v' = v + a*dt)
        self.V += A_total * dt
        self.V = f.normalize(self.V, p=2, dim=1) * self.constant_speed

        # 6. Position Update (p' = p + v'*dt)
        self.P += self.V * dt

        # 7. Boundary Reflection (Handling Non-Exit Cuboid Faces)
        # Iterate over x, y, z dimensions (0, 1, 2)
        for i in range(3):
            min_val = self.bounds[i, 0]
            max_val = self.bounds[i, 1]
            
            # Check for lower bound violation (P < min_val)
            lower_mask = self.P[:, i] < min_val
            if torch.any(lower_mask):
                # Reflect position: p = 2*min - p
                self.P[lower_mask, i] = 2 * min_val - self.P[lower_mask, i]
                # Reflect velocity: v = -v
                self.V[lower_mask, i] = -self.V[lower_mask, i]

            # Check for upper bound violation (P > max_val)
            upper_mask = self.P[:, i] > max_val
            if torch.any(upper_mask):
                # Reflect position: p = 2*max - p
                self.P[upper_mask, i] = 2 * max_val - self.P[upper_mask, i]
                # Reflect velocity: v = -v
                self.V[upper_mask, i] = -self.V[upper_mask, i]
        
        # --- Final Timing ---
        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            current_step_timings['other'] = t_start.elapsed_time(end_event) / 1000.0 # seconds
        else:
            t_end = time.perf_counter()
            current_step_timings['other'] = t_end - t_start
        
        # Accumulate total time
        for key in current_step_timings:
             self.total_timings[key] += current_step_timings[key]
                
        return self.P.detach().cpu().numpy(), current_step_timings