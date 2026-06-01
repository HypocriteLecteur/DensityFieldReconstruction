import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import math
from itertools import cycle
import torch
import os
from matplotlib.animation import FFMpegWriter
from matplotlib.widgets import Slider, CheckButtons, Button 
from dfr.dataset_io import DatasetFactory
from dfr.simulation_config import SimulationConfig
from dfr.camera_system import MultiCameraSystem
from dfr.camera_state import CameraState
from dfr.density_field_model import GaussianModel
from dfr.gaussian_mixture_reduction import GMR

class MultiGMMPlotter:
    def __init__(self, fig=None, ax=None, dpi=300):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = dpi

        self.gmm_data_list = []

        # Set up the figure and 3D axes
        if fig is None or ax is None:
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            # ADDED: Force the subplot to span the entire figure canvas
            # self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
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
        
        if means is not None: gmm_dict['means'] = means
        if covariances is not None: gmm_dict['cov_mats'] = self._transform_covariances(covariances)
        if weights is not None: gmm_dict['weights'] = weights
        if color is not None: gmm_dict['color'] = color
        if label is not None: gmm_dict['label'] = label
        if visible is not None: gmm_dict['visible'] = visible
    
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
        """Draw a single ellipsoid representing a Gaussian component."""
        eigvals, eigvecs = np.linalg.eigh(cov_mat)
        radii = np.sqrt(np.abs(eigvals))
        
        # Increase resolution slightly for print, but keep it reasonable
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        points = np.stack([x*radii[0], y*radii[1], z*radii[2]], axis=0).reshape(3, -1)
        points_rot = np.matmul(eigvecs, points)
        
        x = points_rot[0].reshape(x.shape) + center[0]
        y = points_rot[1].reshape(y.shape) + center[1]
        z = points_rot[2].reshape(z.shape) + center[2]
        
        # Adjust alpha mapping to prevent solid black blobs
        alpha = max(0.05, min(0.6, weight / max_weight)) if max_weight > 0 else 0.1
        
        # Use thinner linewidths for paper to prevent ink bleeding/clutter
        return self.ax.plot_wireframe(
            x, y, z, 
            color=gmm_color, 
            alpha=alpha, 
            rstride=2, cstride=2, 
            linewidth=0.5  # Thinner line for cleaner look
        )

    def draw_camera_frustum(self, camera, color='cyan'):
            """Draws the 3D frustum using the camera's self-contained get_world_frustum method."""
            vertices = camera.state.get_world_frustum()
            
            # Define the 12 edges connecting the 8 frustum corners
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
                [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
                [0, 4], [1, 5], [2, 6], [3, 7]   # Side walls
            ]
            
            plotted_objects = []
            for edge in edges:
                line, = self.ax.plot(
                    vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], 
                    color=color, alpha=0.6, linewidth=1
                )
                plotted_objects.append(line)
                
            # Plot the camera optical center
            center = camera.state.camera_center
            cam_pt = self.ax.scatter(*center, c=color, marker='s', s=30)
            plotted_objects.append(cam_pt)
            
            return plotted_objects
    
    def update(self, real_means=None, cameras=None):
        """
        Draws all managed GMMs and optional auxiliary data.
        Now natively accepts composed Camera objects!
        """
        # Clear previous plot objects
        for obj in self.plot_objects:
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
            # centers = self.ax.scatter(means[:, 0], means[:, 1], means[:, 2], 
            #                          c=gmm_color, marker='o', s=5, label=label_centers)
            # self.plot_objects.append(centers)
            
        # --- Draw Auxiliary Data ---
        if real_means is not None:
            centers_real = self.ax.scatter(real_means[:, 0], real_means[:, 1], real_means[:, 2],
                c='#1f2937',        # Sophisticated dark slate/navy
                s=10,                # Small point size
                alpha=0.65,         # Allow dense areas to visually compound
                edgecolors='none',  
                depthshade=True,    # Crucial for 3D depth perception
                zorder=3            
            )
            self.plot_objects.append(centers_real)
        
        # --- NEW: Draw Cameras ---
        if cameras is not None:
            for cam in cameras:
                frustum_objs = self.draw_camera_frustum(cam)
                self.plot_objects.extend(frustum_objs)

        # 1. Compute dynamic ranges (fall-back)
        auto_x_range, auto_y_range, auto_z_range = self.compute_ranges()
        
        # 2. Use manual range if set, otherwise use automatic range
        final_x_range = self.manual_x_range if self.manual_x_range is not None else auto_x_range
        final_y_range = self.manual_y_range if self.manual_y_range is not None else auto_y_range
        final_z_range = self.manual_z_range if self.manual_z_range is not None else auto_z_range
        
        self.ax.set_xlim(final_x_range)
        self.ax.set_ylim(final_y_range)
        self.ax.set_zlim(final_z_range)
        self.ax.set_box_aspect((
                np.ptp(self.ax.get_xlim()), 
                np.ptp(self.ax.get_ylim()), 
                np.ptp(self.ax.get_zlim())
        ))

        # 4. Enforce equal aspect ratio safely (avoids Matplotlib 3D errors)
        # try:
        #     self.ax.set_box_aspect((
        #         np.ptp(self.ax.get_xlim()), 
        #         np.ptp(self.ax.get_ylim()), 
        #         np.ptp(self.ax.get_zlim())
        #     ))
        # except AttributeError:
        #     pass
            
        # 5. Clean completely for diagram insertion
        # Clean up the "boxy" 3D look by removing the gray panes
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.grid(True, which='major', color='gray', linestyle='-', alpha=0.2)
        # self.ax.grid(True, which='major', linestyle=':', alpha=0.5)
        # self.ax.set_axis_off()

        self.ax.set_xlabel(r'$X$')
        self.ax.set_ylabel(r'$Y$')
        self.ax.set_zlabel(r'$Z$')

        self.fig.tight_layout()

class GMMInteractivePlotter:
    """
    Encapsulates the state and logic for the interactive GMM and Loss history visualization.
    """
    def __init__(self, config_path: str, log_file_path: str, log_file2_path: str=None, 
                 start_step: int=0, end_step=None, step_length: int=1):
        # --- Configuration and Data Setup ---
        
        # Load config and main data
        factory = DatasetFactory()
        self.config = SimulationConfig(config_path)
        self.data = factory.get_dataset(self.config.data_file)
        self.log_file_path = log_file_path
        self.log_file2_path = log_file2_path

        log_data = np.load(os.path.join(self.log_file_path, "statistics.npz"))
        log_data2 = np.load(os.path.join(self.log_file2_path, "statistics.npz"))
        log_data2_baseline = np.load(os.path.join(self.log_file2_path, "statistics_baseline.npz"))
        self.scale_history = log_data['scale']

        # Initialize core components
        self.cam_system = MultiCameraSystem.create_homogeneous_system(
            state_class=CameraState,
            intrinsics=self.config.intrinsics_params,
            H=self.config.H, W=self.config.W, 
            poses_or_RTs=self.config.cam_poses,
            near_clip=self.config.near_clip, far_clip=self.config.far_clip, 
            size=self.config.size,
            device='cuda')

        # --- Plotting State Variables ---
        self.fig = plt.figure(figsize=(20, 10)) # Wider figure for two columns
        self.ax = self.fig.add_subplot(1, 3, 1, projection='3d')
        # History Plot (Right subplot)
        self.loss_2d = self.fig.add_subplot(1, 3, 2)
        self.loss_3d = self.fig.add_subplot(1, 3, 3)

        self.gmm_visualizer = MultiGMMPlotter(fig=self.fig, ax=self.ax)
        self.gmm1_id = None
        self.gmm2_id = None
        self.gmm3_id = None
        self.current_time_step = -1 # Sentinel for history loading
        self.iter_line_2d = None # Line object for current iteration marker
        self.time_line_3d = None

        self.training_history = None
        self.training_history2 = None
        self.training_history2_baseline = None

        # Define limits
        self.MIN_ITER, self.MAX_ITER, self.STEP_ITER = 0, 500-1, 1
        self.MIN_TIME, self.MAX_TIME, self.STEP_TIME = start_step, end_step, step_length

        # Placeholders for widgets
        self.slider_time = None
        self.slider_iter = None
        self.check_buttons = None

        time_steps = np.arange(self.MIN_TIME, self.MAX_TIME, self.STEP_TIME)

        # self.loss_3d.plot(
        #     time_steps,
        #     log_data['final_density_field_loss'], 
        #     color='blue', 
        #     label='3d Loss History'
        # )
        # self.loss_3d.plot(
        #     time_steps,
        #     log_data2['final_density_field_loss'], 
        #     color='orange',
        #     label='3d Loss History 2'
        # )
        # self.loss_3d.plot(
        #     time_steps,
        #     log_data2_baseline['final_density_field_loss'], 
        #     color='black',
        #     label='3d Loss History 2'
        # )
    
    # --- Widget Handler Methods ---
    def _increment_time(self, event):
        new_val = min(self.MAX_TIME, int(self.slider_time.val) + self.STEP_TIME)
        self.slider_time.set_val(new_val)

    def _decrement_time(self, event):
        new_val = max(self.MIN_TIME, int(self.slider_time.val) - self.STEP_TIME)
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
            checkpoint_path2_baseline = os.path.join(self.log_file2_path, f"t_{time_step:03d}", f"baseline_level_0.pth")
            self.training_history = GaussianModel.load_training_history(checkpoint_path)
            self.training_history2 = GaussianModel.load_training_history(checkpoint_path2)
            self.training_history2_baseline = torch.load(checkpoint_path2_baseline, weights_only=False)

            self.current_time_step = time_step
            history_path = os.path.join(self.log_file_path, f"t_{time_step:03d}", "history_level_0.pth")
            history2_path = os.path.join(self.log_file2_path, f"t_{time_step:03d}", "history_level_0.pth")
            
            try:
                loaded_history = torch.load(history_path, weights_only=False)
                loss_history = loaded_history['loss_history']
                loaded_history2 = torch.load(history2_path, weights_only=False)
                loss_history2 = loaded_history2['loss_history']
                
                self.loss_2d.clear()
                iter = self.config.iter
                window_size = 1
                self.loss_2d.plot(
                    np.arange(0, iter, 1),
                    loss_history, 
                    color='k', 
                    label='Loss History'
                )
                self.loss_2d.plot(
                    np.arange(0, iter, 1),
                    loss_history2, 
                    color='k',
                    linestyle='dashed',
                    label='Loss History 2'
                )
                # self.loss_2d.plot(
                #     np.arange(
                #         max(iter * (time_step-window_size+1), 0), 
                #         iter * (time_step+1), 
                #         1
                #         ),
                #     loss_history[-window_size*iter:], 
                #     color='k', 
                #     label='Loss History'
                #     )
                self.loss_2d.set_title(f'Training Loss for Time Step {time_step}')
                self.loss_2d.set_xlabel('Iteration')
                self.loss_2d.set_ylabel('Loss Value')
                self.loss_2d.grid(True, linestyle='--', alpha=0.6)
                
                self.iter_line_2d = None # Reset line object
            except (FileNotFoundError, Exception) as e:
                self.loss_2d.clear()
                self.loss_2d.set_title(f'Loss History (T={time_step}) - File Not Found or Error')
            
            if self.time_line_3d is not None:
                self.time_line_3d.remove()

            self.time_line_3d = self.loss_3d.axvline(
                self.current_time_step, color='r', linestyle='--', linewidth=2, label=f'time {self.current_time_step}'
            )

        # 2. --- Current Iteration Marker Update ---
        if self.iter_line_2d is not None:
            self.iter_line_2d.remove()
            
        if self.loss_2d.lines and iter_val < len(self.loss_2d.lines[0].get_xdata()):
            # Draw a red vertical line/marker at the current iteration
            self.iter_line_2d = self.loss_2d.axvline(
                iter_val, color='r', linestyle='-', linewidth=2, label=f'Iteration {iter_val}'
            )
            # self.iter_line_2d = self.loss_2d.axvline(
            #     time_step*self.config.iter + iter_val, color='r', linestyle='-', linewidth=2, label=f'Iteration {iter_val}'
            # )
            # Re-draw legend to include the vertical line
            self.loss_2d.legend(handles=[self.loss_2d.lines[0], self.iter_line_2d], loc='upper right')

        # 3. --- GMM Visualization Update ---
        # Load scene context data
        scale = self.scale_history[int((time_step - self.MIN_TIME) / self.STEP_TIME)]
        
        # Update the MultiCameraSystem's internal state dynamically!
        try:
            loaded_scene = torch.load(os.path.join(self.log_file_path, f"t_{time_step:03d}", "scene.pth"), weights_only=False)
            
            # Send the new poses directly to the unified State objects
            self.cam_system.cameras[0].state.update_pose(loaded_scene['pose'])
            self.cam_system.cameras[1].state.update_pose(loaded_scene['pose2'])
            
            # Note: We don't even need to extract intrinsics_params because 
            # the Camera objects already know their own intrinsics!
        except FileNotFoundError:
            pass

        # Handle data visibility
        positions = self.data.positions_at_time_step(time_step=time_step)
        _, projections, _, masks = self.cam_system.simulate_vision(positions, is_auto_aim=True, renderer='gaussian')
        is_visible = np.ones((positions.shape[0],), dtype=np.bool)
        for i in range(len(projections)):
            is_visible = is_visible & masks[i]
        real_means_visible = positions[is_visible]
        
        GM_1 = GaussianModel.load_iter(self.training_history, iter_val)
        GM_2 = GaussianModel.load_iter(self.training_history2, iter_val)
        
        # Calculate GMR True data
        r_means, r_weights, r_covs = GMR.kmeans_numpy(
            means=positions[is_visible],
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
        self.training_history2_baseline
        r_means_cpu = self.training_history2_baseline['_xyz'].detach().cpu().numpy()
        r_weights_cpu = self.training_history2_baseline['_weights'].detach().cpu().numpy()
        r_covs_cpu = self.training_history2_baseline['_radius'].detach().cpu().numpy()

        # Initialize or Update GMM data in the visualizer
        if self.gmm1_id is None:
            self.gmm1_id = self.gmm_visualizer.add_gmm(means1, radii1, weights1, color='blue', label='GMM baseline', visible=show_gmm1)
            self.gmm2_id = self.gmm_visualizer.add_gmm(means2, radii2, weights2, color='orange', label='GMM regularization', visible=show_gmm2)
            self.gmm3_id = self.gmm_visualizer.add_gmm(r_means_cpu, r_covs_cpu, r_weights_cpu, color='purple', label='GMR True', visible=show_gmm3)
        else:
            self.gmm_visualizer.update_gmm_data(self.gmm1_id, means=means1, covariances=radii1, weights=weights1, visible=show_gmm1)
            self.gmm_visualizer.update_gmm_data(self.gmm2_id, means=means2, covariances=radii2, weights=weights2, visible=show_gmm2)
            self.gmm_visualizer.update_gmm_data(
                self.gmm3_id, means=r_means_cpu, covariances=r_covs_cpu, 
                weights=r_weights_cpu, visible=show_gmm3
            )

        # Update and redraw the plot
        self.gmm_visualizer.update(
            real_means=real_means_visible, cameras=self.cam_system.cameras[:2]
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

        btn_time_dec = Button(ax_time_dec, f'-{self.STEP_TIME}')
        btn_time_inc = Button(ax_time_inc, f'+{self.STEP_TIME}')
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

def calculate_grid_dims(N):
    """Calculates near-square (Rows, Cols) for N items."""
    if N <= 0:
        return 1, 1
    
    # Calculate the number of columns (C) as the ceiling of sqrt(N)
    C = math.ceil(math.sqrt(N))
    
    # Calculate the number of rows (R)
    R = math.ceil(N / C)
     
    return R, C

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from itertools import cycle

class SimulationVisualizer:
    MODE_3D_ONLY = '3d_only'
    MODE_ALL = 'all'

    def __init__(self, H: int, W: int, cam_num: int, 
                 start_step: int = 0, end_step: int = 100, step_length: int = 1, # Added bounds
                 mode: str='all', save_video: bool=False, 
                 video_filename: str='animation.mp4', fps: int=30, dpi: int=100, 
                 positions_all: np.ndarray=None):
        self.H = H
        self.W = W
        self.cam_num = cam_num
        self.mode = mode.lower()
        
        self.save_video = save_video
        self.fps = fps
        self.dpi = dpi
        self.video_filename = video_filename

        # Ranges for the 3D plot
        self.manual_ranges = {"x": None, "y": None, "z": None}
        if positions_all is not None:
            # Consistent with your previous logic for bounding boxes
            sub = positions_all[::100] if positions_all.shape[0] >= 1000 else positions_all
            self.manual_ranges["x"] = (np.nanmin(sub[..., 0]), np.nanmax(sub[..., 0]))
            self.manual_ranges["y"] = (np.nanmin(sub[..., 1]), np.nanmax(sub[..., 1]))
            self.manual_ranges["z"] = (np.nanmin(sub[..., 2]), np.nanmax(sub[..., 2]))

        # Initialize Figure and Axes
        self.fig, self.axes, self.image_artists = self._setup_plots()
        
        ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(
            ax=ax_slider, 
            label='Time Step', 
            valmin=start_step, 
            valmax=end_step - 1, 
            valinit=start_step, 
            valstep=step_length # Snaps to your step_length automatically
        )

        # Consistent with MultiGMMPlotter: Internal plotter for Gaussians
        # We pass the primary 3D axis to it
        self.gmm_plotter = MultiGMMPlotter(fig=self.fig, ax=self.axes[0])
        if positions_all is not None:
            self.gmm_plotter.set_manual_ranges(
                self.manual_ranges["x"], self.manual_ranges["y"], self.manual_ranges["z"]
            )

        self.writer = FFMpegWriter(fps=fps) if save_video else None
        if save_video:
            self.writer.setup(self.fig, self.video_filename, dpi=dpi)

    def _setup_plots(self):
        """Sets up the layout similar to your existing logic but streamlined."""
        if self.mode == self.MODE_3D_ONLY:
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            return fig, (ax,), None

        # MODE_ALL Layout
        num_2d = self.cam_num
        # Assuming calculate_grid_dims is defined in your scope
        R_2D, C_2D = calculate_grid_dims(num_2d) 
        C_Total = C_2D + 1
        
        fig = plt.figure(figsize=(4 * C_Total, 4 * R_2D))
        gs = GridSpec(nrows=R_2D, ncols=C_Total, figure=fig, width_ratios=[2] + [1]*C_2D)

        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        
        ax2d_list = []
        image_artists = []
        for k in range(num_2d):
            r, c = k // C_2D, (k % C_2D) + 1
            ax_2d = fig.add_subplot(gs[r, c])
            ax_2d.set_title(f"Cam {k}", fontsize=10)
            
            im = ax_2d.imshow(np.zeros((self.H, self.W)), cmap='gray')
            image_artists.append(im)
            ax2d_list.append(ax_2d)

        return fig, (ax_3d, *ax2d_list), image_artists

    def set_interactive_callback(self, render_callback):
        """Hooks the slider to your simulation rendering logic."""
        def on_change(val):
            # val is guaranteed to be a valid time_step based on valstep
            render_callback(int(val))
            self.fig.canvas.draw_idle()
            
        self.slider.on_changed(on_change)

    def run(self, initial_step=None):
        """Starts the visualizer interface."""
        if initial_step is not None:
            self.slider.set_val(initial_step)
        plt.show()

    def update(self, time_step=None, positions=None, cameras=None, imgs=None):
        """Your standard update function."""
        # 1. Update 3D Axis
        self.gmm_plotter.update(real_means=positions, cameras=cameras)
        if time_step is not None:
            self.axes[0].set_title(f'Time step: {time_step}, N = {positions.shape[0]}')

        # 2. Update 2D Images
        if self.mode == self.MODE_ALL and imgs is not None:
            for i, img in enumerate(imgs):
                data = img.cpu().numpy() if hasattr(img, 'cpu') else img
                self.image_artists[i].set_data(data)
                self.image_artists[i].set_clim(vmin=data.min(), vmax=data.max())

        if self.save_video:
            self.writer.grab_frame()