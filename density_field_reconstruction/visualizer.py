import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import math
from itertools import cycle
import torch
import os
from matplotlib.animation import FFMpegWriter
from matplotlib.widgets import Slider, CheckButtons, Button 
from density_field_reconstruction.dataset import DatasetFactory
from density_field_reconstruction.simulation_config import SimulationConfig
from density_field_reconstruction.camera_simulation import MultiCameraSystem
from density_field_reconstruction.density_reconstructor import DensityReconstructor
from density_field_reconstruction.density_field_model import GaussianModel
from density_field_reconstruction.gaussian_mixture_reduction import GMR

class MultiGMMVisualizer:
    def __init__(self, H, W, near_clip=100, far_clip=2000, fig=None, ax=None):
        self.gmm_data_list = []

        # Set up the figure and 3D axes
        if fig is None or ax is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.fig = fig
            self.ax = ax
        self.plot_objects = []
        
        # Get the default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        self._default_colors = cycle(prop_cycle.by_key()['color'])

        self.manual_x_range = None
        self.manual_y_range = None
        self.manual_z_range = None

        self.H = H
        self.W = W
        self.near_clip = near_clip
        self.far_clip = far_clip
    
    def _transform_covariances(self, covariances):
        """Converts isotropic stds to full covariance matrices if needed."""
        covariances = np.asarray(covariances)
        if covariances.ndim == 1:
            # Isotropic standard deviations (N,)
            return np.array([np.eye(3) * (s ** 2) for s in covariances])
        elif covariances.ndim == 2 and covariances.shape[1] == 1:
            # Isotropic standard deviations (N, 1)
            return np.array([np.eye(3) * (s[0] ** 2) for s in covariances])
        elif covariances.ndim == 3 and covariances.shape[1:] == (3, 3):
            # Full covariance matrices (N, 3, 3)
            return covariances
        else:
            raise ValueError("covariances must be (N,), (N, 1), or (N, 3, 3).")
    
    def add_gmm(self, means, covariances, weights, color=None, label=None, visible=True):
        """
        Adds a new GMM to the visualizer's list.
        
        Returns:
            int: The index (ID) of the newly added GMM.
        """
        gmm_dict = {
            'means': means,
            'weights': weights,
            'cov_mats': self._transform_covariances(covariances),
            'color': color,
            'label': label if label is not None else f"GMM {len(self.gmm_data_list) + 1}",
            'visible': visible
        }
        self.gmm_data_list.append(gmm_dict)
        return len(self.gmm_data_list) - 1 # Return the index/ID
    
    def update_gmm_data(self, gmm_id, means=None, covariances=None, weights=None, color=None, label=None, visible=None):
        """
        Updates the parameters of a specific GMM identified by its ID (index).
        
        Parameters:
            gmm_id (int): The index of the GMM in self.gmm_data_list to update.
            means, covariances, weights, color, label: Optional new values.
        """
        if gmm_id < 0 or gmm_id >= len(self.gmm_data_list):
            raise IndexError(f"GMM ID {gmm_id} is out of range. Must be between 0 and {len(self.gmm_data_list) - 1}")
        
        gmm_dict = self.gmm_data_list[gmm_id]
        
        if means is not None:
            gmm_dict['means'] = means
        if covariances is not None:
            gmm_dict['cov_mats'] = self._transform_covariances(covariances)
        if weights is not None:
            gmm_dict['weights'] = weights
        if color is not None:
            gmm_dict['color'] = color
        if label is not None:
            gmm_dict['label'] = label
        if visible is not None:
            gmm_dict['visible'] = visible
    
    def set_manual_ranges(self, x_range=None, y_range=None, z_range=None):
        """
        Manually set the X, Y, and Z axis limits for the plot.
        Pass None to reset a specific axis to automatic calculation.
        """
        self.manual_x_range = x_range
        self.manual_y_range = y_range
        self.manual_z_range = z_range
    
    def compute_ranges(self):
        """
        Compute dynamic plot ranges based on *all* current GMMs.
        """
        if not self.gmm_data_list:
            return (-1, 1), (-1, 1), (-1, 1) # Default if no GMMs
            
        all_means = np.concatenate([gmm['means'] for gmm in self.gmm_data_list if gmm['visible']], axis=0)
        all_cov_mats = np.concatenate([gmm['cov_mats'] for gmm in self.gmm_data_list if gmm['visible']], axis=0)
        
        # Compute max radius (semi-axis) across all Gaussians
        max_radius = 0
        if all_cov_mats.size > 0:
            # Note: We compute eigh once to avoid calling it multiple times for the same cov_mat
            all_radii = [np.max(np.sqrt(np.abs(np.linalg.eigh(cov)[0]))) for cov in all_cov_mats]
            if all_radii:
                 max_radius = np.max(all_radii)
            
        buffer = 2 * max_radius # Buffer proportional to max radius
        
        x_range = (np.min(all_means[:, 0]) - buffer, np.max(all_means[:, 0]) + buffer)
        y_range = (np.min(all_means[:, 1]) - buffer, np.max(all_means[:, 1]) + buffer)
        z_range = (np.min(all_means[:, 2]) - buffer, np.max(all_means[:, 2]) + buffer)
        return x_range, y_range, z_range

    def draw_ellipsoid(self, center, cov_mat, weight, gmm_color, max_weight):
        """
        Draw a single ellipsoid representing a Gaussian component.
        """
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        radii = np.sqrt(np.abs(eigvals))
        
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Scale and Rotate
        points = np.stack([x*radii[0], y*radii[1], z*radii[2]], axis=0).reshape(3, -1)
        points_rot = np.matmul(eigvecs, points)
        
        # Reshape and Translate
        x = points_rot[0].reshape(x.shape) + center[0]
        y = points_rot[1].reshape(y.shape) + center[1]
        z = points_rot[2].reshape(z.shape) + center[2]
        
        # Normalize weight for weights (ensure it's in [0.1, 0.8] for visibility)
        alpha = max(0.1, min(0.8, weight / max_weight)) if max_weight > 0 else 0.1
        
        # Plot wireframe ellipsoid
        return self.ax.plot_wireframe(x, y, z, color=gmm_color, alpha=alpha, rstride=2, cstride=2)

    def update(self, real_means=None, pose1=None, pose2=None, intrinsics_params=None):
        """
        Draws all managed GMMs (using their current data) and optional auxiliary data.
        This function no longer takes GMM parameters as arguments, relying on 
        the data stored in self.gmm_data_list, which must be updated via update_gmm_data().
        """
        # Clear previous plot objects
        for obj in self.plot_objects:
            # Need to check for remove method as plot_frustum might return non-removable objects
            if obj is not None and hasattr(obj, 'remove'):
                if isinstance(obj, list):
                    for obj_ in obj:
                        obj_.remove()
                else:
                    obj.remove()
        self.plot_objects = []
        
        # Reset color cycle
        self._default_colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        
        # --- Draw all GMMs ---
        for i, gmm in enumerate(self.gmm_data_list):
            if not gmm.get('visible', True):
                continue # Skip drawing if the GMM is not visible
            
            # Determine the color: user-defined or next from the default cycle
            gmm_color = gmm['color'] if gmm['color'] else next(self._default_colors)
            
            means, cov_mats, weights, label = gmm['means'], gmm['cov_mats'], gmm['weights'], gmm['label']
            max_weight = np.max(weights) if weights.size > 0 else 1.0

            # Draw ellipsoids for each component
            for j in range(len(means)):
                ellipsoid = self.draw_ellipsoid(
                    means[j], cov_mats[j], weights[j], gmm_color, max_weight
                )
                # text_obj = self.ax.text(means[j][0], means[j][1], means[j][2], f'{j}', 
                #     color=gmm_color, 
                #     fontsize=24, 
                #     ha='center',  # Horizontal alignment
                #     va='bottom')  # Vertical alignment
                # text_obj.set_path_effects([
                #     pe.Stroke(linewidth=4, foreground='black'), # Defines the outline (edge)
                #     pe.Normal()                                 # Draws the original text on top
                # ])
                self.plot_objects.append(ellipsoid)
            
            # Plot Gaussian centers
            # Only add label once to prevent duplicates in the legend
            label_centers = f'{label} Centers' if i == 0 or gmm['color'] is not None else ""
            centers = self.ax.scatter(means[:, 0], means[:, 1], means[:, 2], 
                                     c=gmm_color, marker='o', s=20, label=label_centers)
            self.plot_objects.append(centers)
            
        # --- Draw Auxiliary Data ---
        if real_means is not None:
            centers_real = self.ax.scatter(real_means[:, 0], real_means[:, 1], real_means[:, 2],
                                           c='green', marker='x', s=10, label='Real Centers')
            self.plot_objects.append(centers_real)
        
        if pose1 is not None and intrinsics_params is not None:
            frustum1 = plot_frustum(self.ax, pose1, intrinsics_params, width=self.W, height=self.H, near=self.near_clip, far=self.far_clip, handles=None)
            self.plot_objects.append(frustum1)
        
        if pose2 is not None and intrinsics_params is not None:
            frustum2 = plot_frustum(self.ax, pose2, intrinsics_params, width=self.W, height=self.H, near=self.near_clip, far=self.far_clip, handles=None)
            self.plot_objects.append(frustum2)

        # 1. Compute dynamic ranges (fall-back)
        auto_x_range, auto_y_range, auto_z_range = self.compute_ranges()
        
        # 2. Use manual range if set, otherwise use automatic range
        final_x_range = self.manual_x_range if self.manual_x_range is not None else auto_x_range
        final_y_range = self.manual_y_range if self.manual_y_range is not None else auto_y_range
        final_z_range = self.manual_z_range if self.manual_z_range is not None else auto_z_range
        
        # 3. Apply limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Multi-GMM Visualization')
        self.ax.legend()
        
        self.ax.set_xlim(final_x_range)
        self.ax.set_ylim(final_y_range)
        self.ax.set_zlim(final_z_range)
        self.ax.set_aspect('equal', 'box')
    
    def close(self):
        """Close the plot."""
        plt.close(self.fig)

def compute_fov_frustum(intrinsics, width, height, near=10.0, far=1000.0):
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # Image corners in pixel coordinates
    corners_pixel = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ])
    # Normalized image coordinates
    corners_norm = (corners_pixel - np.array([cx, cy])) / np.array([fx, fy])
    # 3D points at near and far planes
    vertices = []
    for z in [near, far]:
        for norm in corners_norm:
            vertices.append([norm[0] * z, norm[1] * z, z])
    return np.array(vertices)

def frustum_to_world(vertices, cam_pose, R=None, T=None):
    if R is None:
        from density_field_reconstruction.camera_state import CameraState
        P = CameraState.wrd_to_cam(cam_pose)
        vertices_world = []
        for v in vertices:
            v_world = P[:, :3].T @ v - P[:, :3].T @ P[:, 3]
            vertices_world.append(v_world)
        return np.array(vertices_world)
    else:
        vertices_world = []
        for v in vertices:
            v_world = R.T @ v - R.T @ T
            vertices_world.append(v_world)
        return np.array(vertices_world)

def plot_frustum(ax, cam_pose, intrinsics_params, R=None, T=None, width=1000, height=1000, near=100, far=2000, handles=None):
    # Compute frustum vertices in world coordinates
    fov1_vertices = compute_fov_frustum(
        intrinsics_params,
        width=width, height=height, near=near, far=far
    )
    fov1_world = frustum_to_world(fov1_vertices, cam_pose, R=R, T=T)

    # Define edges of the frustum
    fov1_edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]

    # If handles are provided, update them; otherwise, create new ones
    if handles is None:
        handles = []
        for edge in fov1_edges:
            handle = ax.plot3D(fov1_world[edge, 0], fov1_world[edge, 1], fov1_world[edge, 2], 'b', alpha=0.3)
            handles.append(handle[0])
    else:
        for i, edge in enumerate(fov1_edges):
            handles[i].set_data_3d(fov1_world[edge, 0], fov1_world[edge, 1], fov1_world[edge, 2])

    return handles

def plot_cuboid(ax, min_bounds, max_bounds, color='gray', alpha=0.5):
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    vertices = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    for edge in edges:
        ax.plot3D(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], color=color, alpha=alpha)

class GMMInteractivePlotter:
    """
    Encapsulates the state and logic for the interactive GMM and Loss history visualization.
    """
    def __init__(self, config_path: str, log_file_path: str, log_file2_path: str=None):
        # --- Configuration and Data Setup ---
        
        # Load config and main data
        factory = DatasetFactory()
        self.config = SimulationConfig(config_path)
        self.data = factory.get_dataset(self.config.data_file)
        self.positions_all = self.data.trajectories
        self.log_file_path = log_file_path
        self.log_file2_path = log_file2_path

        # Initialize core components
        self.cam_system = MultiCameraSystem.create_system(
            intrinsics=self.config.intrinsics_params,
            H=self.config.H, W=self.config.W, 
            poses=self.config.cam_poses,
            near_clip=self.config.near_clip, far_clip=self.config.far_clip, size=self.config.size)
        self.density_reconstructor = DensityReconstructor(self.config.intrinsics_params, max_iter=self.config.iter)

        # --- Plotting State Variables ---
        self.fig = plt.figure(figsize=(12, 8)) # Wider figure for two columns
        self.ax = self.fig.add_subplot(1, 2, 1, projection='3d')
        # History Plot (Right subplot)
        self.ax_history = self.fig.add_subplot(1, 2, 2)

        self.gmm_visualizer = MultiGMMVisualizer(H=self.config.H, W=self.config.W, 
                                                 near_clip=self.config.near_clip, far_clip=self.config.far_clip,
                                                 fig=self.fig, ax=self.ax)
        self.gmm1_id = None
        self.gmm2_id = None
        self.gmm3_id = None
        self.current_time_step = -1 # Sentinel for history loading
        self.history_line = None # Line object for current iteration marker

        self.training_history = None
        self.training_history2 = None

        # Define limits
        self.MIN_ITER, self.MAX_ITER, self.STEP_ITER = 0, self.config.iter-1, 1
        self.MIN_TIME, self.MAX_TIME, self.STEP_TIME = 0, len(self.positions_all) - 1, 1

        # Placeholders for widgets
        self.slider_time = None
        self.slider_iter = None
        self.check_buttons = None
    
    # --- Widget Handler Methods ---
    def _increment_time(self, event):
        new_val = min(self.MAX_TIME, int(self.slider_time.val) + 1)
        self.slider_time.set_val(new_val)

    def _decrement_time(self, event):
        new_val = max(self.MIN_TIME, int(self.slider_time.val) - 1)
        self.slider_time.set_val(new_val)

    def _increment_iter(self, event):
        new_val = min(self.MAX_ITER, int(self.slider_iter.val) + 1)
        self.slider_iter.set_val(new_val)

    def _decrement_iter(self, event):
        new_val = max(self.MIN_ITER, int(self.slider_iter.val) - 1)
        self.slider_iter.set_val(new_val)
    
    # --- Core Update Logic ---
    def update_plot(self, val):
        """Called when a widget value changes."""
        time_step = int(self.slider_time.val)
        iter_val = int(self.slider_iter.val)
        show_gmm1, show_gmm2, show_gmm3 = self.check_buttons.get_status()

        # 1. --- History Plot Update (Only when time_step changes) ---
        if time_step != self.current_time_step:
            # load training history
            checkpoint_path = os.path.join(self.log_file_path, f"t_{time_step:03d}", f"checkpoint_level_0.pth")
            checkpoint_path2 = os.path.join(self.log_file2_path, f"t_{time_step:03d}", f"checkpoint_level_0.pth")
            self.training_history = GaussianModel.load_training_history(checkpoint_path)
            self.training_history2 = GaussianModel.load_training_history(checkpoint_path2)

            self.current_time_step = time_step
            history_path = os.path.join(self.log_file_path, f"t_{time_step:03d}", "history_level_0.pth")
            history2_path = os.path.join(self.log_file2_path, f"t_{time_step:03d}", "history_level_0.pth")
            
            try:
                loaded_history = torch.load(history_path, weights_only=False)
                loss_history = loaded_history['loss_history']
                loaded_history2 = torch.load(history2_path, weights_only=False)
                loss_history2 = loaded_history2['loss_history']
                
                self.ax_history.clear()
                iter = self.config.iter
                window_size = 1
                self.ax_history.plot(
                    np.arange(0, iter, 1),
                    loss_history, 
                    color='k', 
                    label='Loss History'
                )
                self.ax_history.plot(
                    np.arange(0, iter, 1),
                    loss_history2, 
                    color='k',
                    linestyle='dashed',
                    label='Loss History 2'
                )
                # self.ax_history.plot(
                #     np.arange(
                #         max(iter * (time_step-window_size+1), 0), 
                #         iter * (time_step+1), 
                #         1
                #         ),
                #     loss_history[-window_size*iter:], 
                #     color='k', 
                #     label='Loss History'
                #     )
                self.ax_history.set_title(f'Training Loss for Time Step {time_step}')
                self.ax_history.set_xlabel('Iteration')
                self.ax_history.set_ylabel('Loss Value')
                self.ax_history.grid(True, linestyle='--', alpha=0.6)
                
                self.history_line = None # Reset line object
            except (FileNotFoundError, Exception) as e:
                self.ax_history.clear()
                self.ax_history.set_title(f'Loss History (T={time_step}) - File Not Found or Error')

        # 2. --- Current Iteration Marker Update ---
        if self.history_line is not None:
            self.history_line.remove()
            
        if self.ax_history.lines and iter_val < len(self.ax_history.lines[0].get_xdata()):
            # Draw a red vertical line/marker at the current iteration
            self.history_line = self.ax_history.axvline(
                iter_val, color='r', linestyle='-', linewidth=2, label=f'Iteration {iter_val}'
            )
            # self.history_line = self.ax_history.axvline(
            #     time_step*self.config.iter + iter_val, color='r', linestyle='-', linewidth=2, label=f'Iteration {iter_val}'
            # )
            # Re-draw legend to include the vertical line
            self.ax_history.legend(handles=[self.ax_history.lines[0], self.history_line], loc='upper right')

        # 3. --- GMM Visualization Update ---
        # Load scene context data
        pose, pose2, intrinsics_params = np.eye(4), np.eye(4), np.array([500, 500, 500, 500])
        scale = 1.0
        try:
            loaded_scene = torch.load(os.path.join(self.log_file_path, f"t_{time_step:03d}", "scene.pth"), weights_only=False)
            pose = loaded_scene['pose']
            pose2 = loaded_scene['pose2']
            intrinsics_params = loaded_scene['intrinsics_params']
            scale = loaded_scene['scale']
        except FileNotFoundError:
            pass

        # Handle data visibility
        _, projections, _ = self.cam_system.simulate_vision(self.positions_all[time_step], renderer='gaussian')
        is_visible = np.zeros((self.positions_all[time_step].shape[0],), dtype=np.bool)
        for i in range(len(projections)):
            projection = projections[i]
            is_visible_ = (projection[:, 0] > 0).squeeze() & (projection[:, 1] > 0).squeeze() & \
                (projection[:, 0] < self.config.H).squeeze() & (projection[:, 1] < self.config.W).squeeze()
            is_visible = is_visible | is_visible_
        real_means_visible = self.positions_all[time_step][is_visible]
        
        try:
            GM_1 = GaussianModel.load_iter(self.training_history, iter_val)
            GM_2 = GaussianModel.load_iter(self.training_history2, iter_val)
        except Exception:
            self.gmm_visualizer.clear_data()
            self.fig.canvas.draw_idle()
            return
        
        # Calculate GMR True data
        r_means, r_weights, r_covs = GMR.kmeans_numpy(
            means=self.positions_all[time_step][is_visible],
            sigma=scale,
            cluster_size=GM_1._xyz.shape[0]
        )

        # Data preparation
        means1 = GM_1._xyz.detach().cpu().numpy()
        radii1 = GM_1._radius.detach().cpu().numpy()
        weights1 = GM_1._weights.detach().cpu().numpy()
        means2 = GM_2._xyz.detach().cpu().numpy()
        radii2 = GM_2._radius.detach().cpu().numpy()
        weights2 = GM_2._weights.detach().cpu().numpy()
        r_means_cpu = r_means.detach().cpu().numpy()
        r_weights_cpu = r_weights.detach().cpu().numpy()
        r_covs_cpu = r_covs.detach().cpu().numpy()

        # Initialize or Update GMM data in the visualizer
        if self.gmm1_id is None:
            self.gmm1_id = self.gmm_visualizer.add_gmm(means1, radii1, weights1, color='blue', label='GMM baseline', visible=show_gmm1)
            self.gmm2_id = self.gmm_visualizer.add_gmm(means2, radii2, weights2, color='orange', label='GMM regularization', visible=show_gmm2)
            self.gmm3_id = self.gmm_visualizer.add_gmm(r_means_cpu, r_weights_cpu, r_covs_cpu, color='purple', label='GMR True', visible=show_gmm3)
        else:
            self.gmm_visualizer.update_gmm_data(self.gmm1_id, means=means1, covariances=radii1, weights=weights1, visible=show_gmm1)
            self.gmm_visualizer.update_gmm_data(self.gmm2_id, means=means2, covariances=radii2, weights=weights2, visible=show_gmm2)
            self.gmm_visualizer.update_gmm_data(
                self.gmm3_id, means=r_means_cpu, covariances=r_covs_cpu, 
                weights=r_means_cpu, visible=show_gmm3
            )

        # Update and redraw the plot
        intrinsics_np = intrinsics_params.detach().cpu().numpy() if isinstance(intrinsics_params, torch.Tensor) else intrinsics_params
        
        self.gmm_visualizer.update(
            real_means=real_means_visible, pose1=pose, pose2=pose2, intrinsics_params=intrinsics_np
        )
        self.gmm_visualizer.ax.set_title(f'Timestep {time_step} Iteration {iter_val}')
        
        self.fig.canvas.draw_idle()

    def run(self):
        """Sets up the Matplotlib GUI and starts the interactive loop."""        
        # Adjust layout for controls at the bottom
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.25) 
        
        # 2. Define Axes for Widgets (using normalized figure coordinates)
        # Note: We use the bottom space for all controls
        
        # TIME CONTROLS (Top row of controls)
        ax_time_dec = self.fig.add_axes([0.05, 0.15, 0.04, 0.04]) # Time -1 button
        ax_time_inc = self.fig.add_axes([0.09, 0.15, 0.04, 0.04]) # Time + button
        ax_time = self.fig.add_axes([0.2, 0.15, 0.50, 0.04]) # Time slider
        
        # ITERATION CONTROLS (Middle row of controls)
        ax_iter_dec = self.fig.add_axes([0.05, 0.1, 0.04, 0.04]) # Iter -1 button
        ax_iter_inc = self.fig.add_axes([0.09, 0.1, 0.04, 0.04]) # Iter + button
        ax_iter = self.fig.add_axes([0.2, 0.1, 0.50, 0.04]) # Iter slider
        
        # CHECKBOXES (Bottom row of controls)
        ax_check = self.fig.add_axes([0.75, 0.1, 0.2, 0.09]) 

        # 3. Create Widgets
        self.slider_time = Slider(ax=ax_time, label='T:', valmin=self.MIN_TIME, valmax=self.MAX_TIME, valinit=self.MIN_TIME, valstep=self.STEP_TIME, valfmt='%d')
        self.slider_iter = Slider(ax=ax_iter, label='Iter:', valmin=self.MIN_ITER, valmax=self.MAX_ITER, valinit=self.MIN_ITER, valstep=self.STEP_ITER, valfmt='%d')

        btn_time_dec = Button(ax_time_dec, '-1')
        btn_time_inc = Button(ax_time_inc, '+1')
        btn_iter_dec = Button(ax_iter_dec, '-1')
        btn_iter_inc = Button(ax_iter_inc, '+1')

        labels = ['GMM Baseline (Blue)', 'GMM Alt. (Orange)', 'GMR True (Purple)']
        self.check_buttons = CheckButtons(ax=ax_check, labels=labels, actives=[True, False, False])

        # 4. Connect Widgets to Handlers
        self.slider_time.on_changed(self.update_plot)
        self.slider_iter.on_changed(self.update_plot)
        self.check_buttons.on_clicked(self.update_plot)

        btn_time_dec.on_clicked(self._decrement_time)
        btn_time_inc.on_clicked(self._increment_time)
        btn_iter_dec.on_clicked(self._decrement_iter)
        btn_iter_inc.on_clicked(self._increment_iter)

        # 5. Initial Run and Display
        self.update_plot(None)
        plt.show()

def calculate_grid_dims(N):
    """Calculates near-square (Rows, Cols) for N items."""
    if N <= 0:
        return 1, 1
    
    # Calculate the number of columns (C) as the ceiling of sqrt(N)
    C = math.ceil(math.sqrt(N))
    
    # Calculate the number of rows (R)
    R = math.ceil(N / C)
     
    return R, C

class SimulationVisualizer:
    # Modes for visualization:
    MODE_3D_ONLY = '3d_only'  # Use only the first (3D) axis
    MODE_ALL = 'all'          # Use all three axes (3D + 2 camera views)

    def __init__(self, 
                 intrinsics_params: np.ndarray, 
                 H: int, W: int, 
                 cam_num: int, 
                 mode: str=MODE_ALL, 
                 save_video: bool=False, 
                 video_filename: str='animation.mp4', 
                 fps: int=30, dpi: int=100, 
                 positions_all: np.ndarray=None):
        self.intrinsics_params = intrinsics_params
        self.H = H
        self.W = W

        self.cam_num = cam_num
        self.mode = mode.lower()
        if self.mode not in [self.MODE_3D_ONLY, self.MODE_ALL]:
             raise ValueError(f"Mode must be '{self.MODE_3D_ONLY}' or '{self.MODE_ALL}'")
             
        self.save_video = save_video
        self.fps = fps
        self.dpi = dpi
        self.video_filename = video_filename

        if positions_all is None:
            self.min_positions = None
            self.max_positions = None
        else:
            self.min_positions = np.min(positions_all, axis=(0, 1))
            self.max_positions = np.max(positions_all, axis=(0, 1))
            
        self.fig, self.axes, self.image_artists = self._setup_plots()
        
        self.writer = FFMpegWriter(fps=fps) if save_video else None
        if save_video:
            # Use the new video_filename
            self.writer.setup(self.fig, self.video_filename, dpi=dpi) 

    def _setup_plots(self):
        """Sets up the matplotlib figure and axes based on the selected mode."""
        image_artists = None

        if self.mode == self.MODE_3D_ONLY:
            fig = plt.figure(figsize=(8, 8)) # Use a squarer figure for a single plot
            # Single 3D subplot
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_axis_off()

            axes = (ax,) # Tuple containing only the 3D axis
            
        elif self.mode == self.MODE_ALL:
            num_2d_axes = self.cam_num
            R_2D, C_2D = calculate_grid_dims(num_2d_axes)

            R_Total = R_2D
            C_Total = C_2D + 1 # +1 for the 3D plot

            fig = plt.figure(figsize=(4 * C_Total, 4 * R_Total))

            width_ratios = [2] + [1] * C_2D
            gs = GridSpec(nrows=R_Total, ncols=C_Total, figure=fig, width_ratios=width_ratios)

            ax = fig.add_subplot(gs[:, 0], projection='3d')
            ax.set_title("3D Subplot (Spans All Rows)")
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            ax2d = []
            image_artists = []
            for k in range(num_2d_axes):
                # Calculate the row and column index within the right-side grid (starting at column 1)
                row_idx = k // C_2D
                col_idx = (k % C_2D) + 1 # +1 because column 0 is for the 3D plot
                
                # If we run out of rows, stop (shouldn't happen with the ceiling calculation, but safe check)
                if row_idx >= R_2D:
                    break 
                    
                ax_2d = fig.add_subplot(gs[row_idx, col_idx])
                ax_2d.set_title(f"2D Plot {k+1}", fontsize=10)
                ax2d.append(ax_2d)

                initial_data = np.zeros((self.H, self.W))
                im = ax_2d.imshow(initial_data, cmap='gray')
                
                im.set_clim(vmin=0, vmax=20)
                
                image_artists.append(im)

            axes = (ax, *ax2d)

        # plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        return fig, axes, image_artists


    def update(self, time_step=None, positions=None,
               cam_poses=None, imgs=None, near_clip=5, far_clip=300):
        """Updates the plots with new data."""
        
        # Determine which axes are available based on the mode
        ax = self.axes[0]
        
        # --- 3D Plot Update (ax) ---
        if time_step is not None or positions is not None or cam_poses is not None:
            ax.clear()
            # The cuboid is plotted only if bounds were initialized in __init__
            if self.min_positions is not None and self.max_positions is not None:
                plot_cuboid(ax, self.min_positions, self.max_positions, color='black', alpha=0.5)
            
            # Plot positions if provided
            if positions is not None:
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, c='blue')
            
            # Plot camera frustums if provided
            if cam_poses is not None:
                for cam_pose in cam_poses:
                    plot_frustum(ax, cam_pose, self.intrinsics_params, far=far_clip, near=near_clip)
            
            ax.set_aspect('equal', 'box')
            
            # Update title
            if time_step is not None:
                ax.set_title(f'Time step: {time_step}')
            else:
                ax.set_title('3D View')
        
        # The following image updates only run if in MODE_ALL
        if self.mode == self.MODE_ALL:
            ax, *ax_2d = self.axes
            for i in range(len(ax_2d)):
                img_data = imgs[i].cpu() if hasattr(imgs[i], 'cpu') else imgs[i]
                self.image_artists[i].set_clim(vmin=0, vmax=torch.max(imgs[i]))
                self.image_artists[i].set_data(img_data)
        
        # ax.set_axis_off()

        if self.save_video:
            self.writer.grab_frame()
        plt.pause(0.001)

    def close(self):
        """Finalizes the video saving and closes the plot."""
        if self.save_video:
            self.writer.finish()
        plt.close(self.fig)