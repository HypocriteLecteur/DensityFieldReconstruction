import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Scene Setup ---
MAX_POINTS = 50
# Generate a fixed pool of i.i.d uniform points in [-0.5, 0.5]^3
# We slice this array based on the slider so points don't jump around during updates
all_true_points = (torch.rand((MAX_POINTS, 3), device=device) - 0.5)

def get_camera_matrices(num_cameras, radius=2.5):
    """Generate camera matrices evenly distributed on the XY equator."""
    matrices = []
    angles = np.linspace(0, 2*np.pi, num_cameras, endpoint=False)
    for angle in angles:
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)
        pos = torch.tensor([cx, cy, 0.0], dtype=torch.float32, device=device)
        
        z_axis = -pos / torch.norm(pos)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        x_axis = torch.linalg.cross(up, z_axis)
        x_axis = x_axis / torch.norm(x_axis)
        y_axis = torch.linalg.cross(z_axis, x_axis)
        
        R = torch.stack([x_axis, y_axis, z_axis])
        t = -R @ pos
        P = torch.cat([R, t.unsqueeze(1)], dim=1)
        matrices.append(P)
    return torch.stack(matrices)

# --- 2. Voxel Carving Core ---
def compute_visual_hull(current_points, num_cameras, radius_tol, density, max_voxels=1_000_000):
    density = min(density, int(max_voxels**(1/3)))
    
    linspace = torch.linspace(-0.6, 0.6, density, device=device)
    X, Y, Z = torch.meshgrid(linspace, linspace, linspace, indexing='ij')
    voxels = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=1)
    
    cams = get_camera_matrices(num_cameras)
    
    voxels_h = torch.cat([voxels, torch.ones((voxels.shape[0], 1), device=device)], dim=1)
    points_h = torch.cat([current_points, torch.ones((current_points.shape[0], 1), device=device)], dim=1)
    
    valid_mask = torch.ones(voxels.shape[0], dtype=torch.bool, device=device)
    batch_size = 50000 
    
    for cam_idx in range(num_cameras):
        P = cams[cam_idx]
        
        proj_pts = (P @ points_h.T).T
        proj_pts_2d = proj_pts[:, :2] / proj_pts[:, 2:3]
        
        for i in range(0, voxels.shape[0], batch_size):
            if not valid_mask[i:i+batch_size].any():
                continue
            
            v_batch = voxels_h[i:i+batch_size]
            proj_v = (P @ v_batch.T).T
            
            front_mask = proj_v[:, 2] > 0
            proj_v_2d = proj_v[:, :2] / proj_v[:, 2:3]
            
            dists = torch.norm(proj_v_2d.unsqueeze(1) - proj_pts_2d.unsqueeze(0), dim=2)
            min_dists = dists.min(dim=1)[0]
            
            batch_valid = (min_dists <= radius_tol) & front_mask
            valid_mask[i:i+batch_size] &= batch_valid
            
    return voxels[valid_mask].cpu().numpy()

# --- 3. Interactive Visualization ---
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
# Increased bottom margin to fit 4 sliders
plt.subplots_adjust(left=0.15, bottom=0.35)

def update_plot(val):
    ax.clear()
    n_cams = int(slider_cams.val)
    r_tol = slider_radius.val
    dense = int(slider_density.val)
    n_pts = int(slider_points.val)
    
    # Slice the active points
    current_points = all_true_points[:n_pts]
    pts_cpu = current_points.cpu().numpy()
    
    hull_voxels = compute_visual_hull(current_points, n_cams, r_tol, dense)
    
    ax.scatter(pts_cpu[:, 0], pts_cpu[:, 1], pts_cpu[:, 2], c='red', s=50, label='True Points', depthshade=False)
    
    if len(hull_voxels) > 0:
        ax.scatter(hull_voxels[:, 0], hull_voxels[:, 1], hull_voxels[:, 2], c='blue', s=10, alpha=0.3, label='Visual Hull', marker='s')
    
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.6, 0.6])
    ax.set_zlim([-0.6, 0.6])
    ax.set_title(f"Visual Hull (Cameras: {n_cams}, Points: {n_pts}, Voxels: {len(hull_voxels)})")
    ax.legend()
    fig.canvas.draw_idle()

# Sliders setup
axcolor = 'lightgoldenrodyellow'
ax_cams = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_rad  = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_den  = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_pts  = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

slider_cams = Slider(ax_cams, 'Cameras', 3, 10, valinit=3, valstep=1)
slider_radius = Slider(ax_rad, 'Radius Tol', 0.01, 0.3, valinit=0.1)
slider_density = Slider(ax_den, 'Density', 10, 80, valinit=30, valstep=1)
slider_points = Slider(ax_pts, 'Point Count', 1, MAX_POINTS, valinit=15, valstep=1)

slider_cams.on_changed(update_plot)
slider_radius.on_changed(update_plot)
slider_density.on_changed(update_plot)
slider_points.on_changed(update_plot)

update_plot(None)
plt.show()