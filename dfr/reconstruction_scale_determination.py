import numpy as np
import torch
from numba import njit
from dfr.camera_system import Camera
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import cv2
from experiments.power_law import move_figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from typing import Union, Tuple, Optional, Any
import plotly.graph_objects as go

import matplotlib.colors as mcolors
import matplotlib.cm as cm

def visualize_gmm_matplotlib_clean(
    gmm_mean, 
    gmm_radius, 
    gmm_weights,
    cmap_name: str = 'plasma',
    alpha: float = 0.4,
    dpi: int = 300
):
    """
    Visualizes 3D Isotropic Gaussian Mixture Models using Matplotlib.
    Formatted for clean diagram insertion (no axes or background).
    
    Args:
        gmm_mean: GPU/CPU tensor of shape [N, 3]
        gmm_radius: GPU/CPU tensor of shape [N, 1]
        gmm_weights: GPU/CPU tensor of shape [N, 1]
        cmap_name: Matplotlib colormap string.
        alpha: Transparency of the spheres.
        dpi: Resolution for print export.
        
    Returns:
        fig, ax: The generated Matplotlib figure and 3D axis.
    """
    # --- 1. Data Prep (Handle GPU Tensors) ---
    def to_numpy(t):
        if hasattr(t, 'detach'):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    means = to_numpy(gmm_mean)              # [N, 3]
    radii = to_numpy(gmm_radius).ravel()    # [N]
    weights = to_numpy(gmm_weights).ravel() # [N]
    
    # --- 2. Setup Color Mapping ---
    w_min, w_max = weights.min(), weights.max()
    if np.isclose(w_min, w_max):
        norm = mcolors.Normalize(vmin=w_min - 0.1, vmax=w_max + 0.1)
    else:
        norm = mcolors.Normalize(vmin=w_min, vmax=w_max)
        
    try:
        cmap = cm.colormaps[cmap_name]
    except AttributeError:
        cmap = cm.get_cmap(cmap_name)

    # --- 3. Setup Base Sphere Math ---
    # Kept relatively low-res (20x10) to prevent Matplotlib from freezing on large N
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    
    base_x = np.outer(np.cos(u), np.sin(v))
    base_y = np.outer(np.sin(u), np.sin(v))
    base_z = np.outer(np.ones(np.size(u)), np.cos(v))

    # --- 4. Plotting ---
    fig = plt.figure(figsize=(6, 6), dpi=dpi, layout='tight')
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(means)):
        # Scale and translate the base sphere
        X = base_x * radii[i] + means[i, 0]
        Y = base_y * radii[i] + means[i, 1]
        Z = base_z * radii[i] + means[i, 2]
        
        # Get color based on weight
        color = cmap(norm(weights[i]))
        
        # Plot the surface
        ax.plot_surface(
            X, Y, Z, 
            color=color, 
            alpha=alpha, 
            linewidth=0,      # Removes wireframe lines for a smoother look
            antialiased=True,
            shade=True        # Enables basic Matplotlib lighting
        )

    # --- 5. Clean Bounds & Aspect Ratio ---
    # We must calculate manual bounds since plot_surface doesn't auto-scale perfectly
    max_radius = radii.max() if len(radii) > 0 else 0
    x_min, x_max = means[:, 0].min() - max_radius, means[:, 0].max() + max_radius
    y_min, y_max = means[:, 1].min() - max_radius, means[:, 1].max() + max_radius
    z_min, z_max = means[:, 2].min() - max_radius, means[:, 2].max() + max_radius

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # Enforce equal aspect ratio to keep spheres perfectly round
    try:
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
    except AttributeError:
        pass 

    # Completely remove all axes, grids, and background panes
    ax.set_axis_off()

    return fig, ax

def visualize_voxel_ellipsoid_plotly(
    positions: Union[np.ndarray, Any], 
    grid_size: Union[np.ndarray, list, tuple], 
    aabb: Union[np.ndarray, Any],
    voxel_color: str = '#1f77b4',
    voxel_alpha: float = 0.5
) -> go.Figure:
    """
    Visualizes a reconstructed visual hull as explicit 3D voxels inside an AABB using Plotly.
    """
    # --- 1. Data Prep & Voxel Math ---
    if hasattr(positions, 'cpu'):
        positions = positions.cpu().numpy()
    if hasattr(aabb, 'cpu'):
        aabb = aabb.cpu().numpy()
        
    positions = np.asarray(positions)
    aabb = np.asarray(aabb)
    grid_size = np.asarray(grid_size)

    voxel_sizes = (aabb[:, 1] - aabb[:, 0]) / grid_size
    dx, dy, dz = voxel_sizes / 2.0 

    # --- 2. Generate the AABB Wireframe ---
    x_min, x_max = aabb[0]
    y_min, y_max = aabb[1]
    z_min, z_max = aabb[2]
    x_tmp = (x_max - x_min)
    x_min = x_min + x_tmp / 5
    x_max = x_max - x_tmp / 5

    y_tmp = (y_max - y_min)
    y_min = y_min + y_tmp / 5
    y_max = y_max - y_tmp / 5

    z_tmp = (z_max - z_min)
    z_min = z_min + z_tmp / 3
    z_max = z_max - z_tmp / 3

    box_corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min], 
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]  
    ])

    lines_idx = [
        0, 1, 2, 3, 0,  
        4, 5, 6, 7, 4,  
        -1, 1, 5, -1, 2, 6, -1, 3, 7 
    ]
    
    box_x, box_y, box_z = [], [], []
    for idx in lines_idx:
        if idx == -1:
            # Plotly uses None to break lines instead of np.nan
            box_x.append(None); box_y.append(None); box_z.append(None) 
        else:
            box_x.append(box_corners[idx, 0])
            box_y.append(box_corners[idx, 1])
            box_z.append(box_corners[idx, 2])

    # --- 3. Generate Voxel Mesh (Triangulated for Plotly) ---
    vertices = []
    i, j, k = [], [], []
    vertex_offset = 0

    for center in positions:
        x, y, z = center
        # 8 vertices of the current voxel
        v = [
            [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz], [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
            [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz], [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]
        ]
        vertices.extend(v)

        # 12 Triangles (2 per face) to build the cube
        triangles = [
            (0, 1, 2), (0, 2, 3), # bottom
            (4, 5, 6), (4, 6, 7), # top
            (0, 1, 5), (0, 5, 4), # front
            (2, 3, 7), (2, 7, 6), # back
            (1, 2, 6), (1, 6, 5), # right
            (0, 3, 7), (0, 7, 4)  # left
        ]

        # Shift indices by the current vertex offset
        for tri in triangles:
            i.append(tri[0] + vertex_offset)
            j.append(tri[1] + vertex_offset)
            k.append(tri[2] + vertex_offset)

        vertex_offset += 8

    vertices = np.array(vertices)

    # --- 4. Plotting ---
    fig = go.Figure()

    # Add Voxels as a single Mesh3d object
    if len(vertices) > 0:
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=i, j=j, k=k,
            color=voxel_color,
            opacity=voxel_alpha,
            flatshading=True, # Essential for the "blocky" voxel look
            hoverinfo='skip',
            name='Admissible Voxels'
        ))

    # Add AABB Wireframe
    fig.add_trace(go.Scatter3d(
        x=box_x, y=box_y, z=box_z,
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        hoverinfo='skip',
        name='AABB Boundary'
    ))

    # --- 5. Clean Layout for Diagrams ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), # Hide axes for clean diagrams
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data' # Enforces equal aspect ratio
        ),
        margin=dict(l=0, r=0, b=0, t=0), # Remove whitespace
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.write_html("voxel_visualization.html")

    return fig

def visualize_voxel_ellipsoid_mpl(
    positions: Union[np.ndarray, Any], 
    grid_size: Union[np.ndarray, list, tuple], 
    aabb: Union[np.ndarray, Any],
    voxel_color: str = '#1f77b4',
    voxel_alpha: float = 0.1,
    dpi: int = 300
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualizes a reconstructed visual hull as explicit 3D voxels inside an AABB.
    Stripped of all text, axes, and legends for insertion into workflow diagrams.
    """
    # --- 1. Data Prep & Voxel Math ---
    if hasattr(positions, 'cpu'):
        positions = positions.cpu().numpy()
    if hasattr(aabb, 'cpu'):
        aabb = aabb.cpu().numpy()
        
    positions = np.asarray(positions)
    aabb = np.asarray(aabb)
    grid_size = np.asarray(grid_size)

    voxel_sizes = (aabb[:, 1] - aabb[:, 0]) / grid_size
    dx, dy, dz = voxel_sizes / 2.0 

    # --- 2. Generate the AABB Wireframe ---
    x_min, x_max = aabb[0]
    y_min, y_max = aabb[1]
    z_min, z_max = aabb[2]

    x_tmp = (x_max - x_min)
    x_min = x_min + x_tmp / 5
    x_max = x_max - x_tmp / 5

    y_tmp = (y_max - y_min)
    y_min = y_min + y_tmp / 5
    y_max = y_max - y_tmp / 5

    z_tmp = (z_max - z_min)
    z_min = z_min + z_tmp / 3
    z_max = z_max - z_tmp / 3

    box_corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min], 
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]  
    ])

    lines_idx = [
        0, 1, 2, 3, 0,  
        4, 5, 6, 7, 4,  
        -1, 1, 5, -1, 2, 6, -1, 3, 7 
    ]
    
    box_x, box_y, box_z = [], [], []
    for idx in lines_idx:
        if idx == -1:
            box_x.append(np.nan); box_y.append(np.nan); box_z.append(np.nan) 
        else:
            box_x.append(box_corners[idx, 0])
            box_y.append(box_corners[idx, 1])
            box_z.append(box_corners[idx, 2])

    # --- 3. Generate Voxel Faces ---
    faces = []
    for center in positions:
        x, y, z = center
        v = [
            [x-dx, y-dy, z-dz], [x+dx, y-dy, z-dz], [x+dx, y+dy, z-dz], [x-dx, y+dy, z-dz],
            [x-dx, y-dy, z+dz], [x+dx, y-dy, z+dz], [x+dx, y+dy, z+dz], [x-dx, y+dy, z+dz]
        ]
        faces.extend([
            [v[0], v[1], v[2], v[3]], 
            [v[4], v[5], v[6], v[7]], 
            [v[0], v[1], v[5], v[4]], 
            [v[2], v[3], v[7], v[6]], 
            [v[1], v[2], v[6], v[5]], 
            [v[0], v[3], v[7], v[4]]  
        ])

    # --- 4. Plotting (Diagram Optimized) ---
    # Using layout='tight' to crop out excess whitespace around the diagram
    fig = plt.figure(figsize=(6, 6), dpi=dpi, layout='tight')
    ax = fig.add_subplot(111, projection='3d')

    # Add AABB
    # ax.plot(box_x, box_y, box_z, color='black', linewidth=1.0, linestyle='--', zorder=10)

    # Add Voxels 
    voxel_collection = Poly3DCollection(
        faces, 
        alpha=voxel_alpha, 
        facecolors=voxel_color, 
        edgecolors='black', 
        linewidths=0.2  
    )
    ax.add_collection3d(voxel_collection)

    # --- 5. Clean Bounds & Aspect Ratio ---
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    try:
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
    except AttributeError:
        pass 

    # Completely remove all axes, grids, background panes, and labels
    ax.set_axis_off()

    return fig, ax

@njit
def get_plane(p1, p2, p3):
    """Calculates plane equation Ax + By + Cz + D = 0 where interior is <= 0."""
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm > 1e-9:
        n = n / norm
    d = -np.dot(n, p1)
    # Result is a 4-element array [A, B, C, D]
    res = np.empty(4)
    res[0], res[1], res[2], res[3] = n[0], n[1], n[2], d
    return res

@njit
def get_frustum_planes_numba(v):
    """
    Converts 8 frustum vertices to 6 inward-facing plane equations.
    Assumes vertices are ordered: 0-3 near (TL, TR, BR, BL), 4-7 far (TL, TR, BR, BL)
    """
    planes = np.empty((6, 4))
    # Near plane (0, 1, 2)
    planes[0] = get_plane(v[0], v[2], v[1])
    # Far plane (4, 7, 6)
    planes[1] = get_plane(v[4], v[6], v[7])
    # Left (0, 3, 7)
    planes[2] = get_plane(v[0], v[7], v[3])
    # Right (1, 5, 2)
    planes[3] = get_plane(v[1], v[2], v[5])
    # Top (0, 4, 5)
    planes[4] = get_plane(v[0], v[5], v[4])
    # Bottom (3, 2, 6)
    planes[5] = get_plane(v[3], v[6], v[2])
    return planes

@njit
def clip_polygon_by_plane(poly_verts, plane):
    num_verts = len(poly_verts)
    if num_verts == 0:
        return np.zeros((0, 3))
    
    # Resulting polygon can have at most num_verts + 1 vertices per plane clip
    new_verts = np.zeros((num_verts * 2, 3))
    count = 0
    
    n = plane[:3]
    d = plane[3]
    
    for i in range(num_verts):
        p1 = poly_verts[i]
        p2 = poly_verts[(i + 1) % num_verts]
        
        dist1 = np.dot(n, p1) + d
        dist2 = np.dot(n, p2) + d
        
        if dist2 <= 0:
            if dist1 > 0:
                t = dist1 / (dist1 - dist2)
                new_verts[count] = p1 + t * (p2 - p1)
                count += 1
            new_verts[count] = p2
            count += 1
        elif dist1 <= 0:
            t = dist1 / (dist1 - dist2)
            new_verts[count] = p1 + t * (p2 - p1)
            count += 1
                
    return new_verts[:count]

@njit
def compute_frustum_intersection_numba(f1_verts, f2_planes):
    # Faces defined by vertex indices
    faces_idx = np.array([
        [0, 1, 2, 3], [4, 7, 6, 5], # Near, Far
        [0, 3, 7, 4], [1, 5, 6, 2], # Left, Right
        [0, 4, 5, 1], [3, 2, 6, 7]  # Top, Bottom
    ])
    
    # Buffer to hold all vertices of the resulting polytope
    result_buffer = np.zeros((128, 3))
    total_count = 0
    
    for f_i in range(6):
        # Build initial face (4 vertices)
        poly = np.empty((4, 3))
        for v_i in range(4):
            poly[v_i] = f1_verts[faces_idx[f_i, v_i]]
            
        # Clip the face against all 6 planes of the other frustum
        for p_i in range(6):
            poly = clip_polygon_by_plane(poly, f2_planes[p_i])
            if len(poly) == 0:
                break
        
        # Collect vertices
        for p_idx in range(len(poly)):
            if total_count < 128:
                result_buffer[total_count] = poly[p_idx]
                total_count += 1
                
    return result_buffer[:total_count]

def get_aabb(cameras: list[Camera]):
    fov1_world = cameras[0].state.get_world_frustum()
    fov2_world = cameras[1].state.get_world_frustum()

    fov2_planes = get_frustum_planes_numba(fov2_world)
    intersection = compute_frustum_intersection_numba(fov1_world, fov2_planes)

    aabb = np.vstack((np.min(intersection, axis=0), np.max(intersection, axis=0))).T
    return aabb

def extract_dilated_masks(images, intensity_threshold=0.05, dilation_iters=2):
    """
    Creates conservative 2D masks by thresholding and dilating.
    GPU-only — uses max-pool to approximate morphological dilation.
    """
    masks = []
    # kernel_size matches OpenCV's 11×11 ellipse applied `dilation_iters` times
    k = 11  # matches original cv2 kernel size

    for img in images:
        binary = (img > intensity_threshold).float().unsqueeze(0).unsqueeze(0)
        for _ in range(dilation_iters):
            binary = torch.nn.functional.max_pool2d(
                binary, kernel_size=k, stride=1, padding=k // 2)
        masks.append(binary.squeeze().bool())
    return masks
# def extract_dilated_masks(images, intensity_threshold=0.05, dilation_iters=2):
#     """
#     Creates conservative 2D masks by thresholding and dilating.
#     """
#     device = images[0].device
#     masks = []
    
#     for img in images:
#         img_min = img.min()
#         img_max = img.max()
#         # img_norm = (img - img_min) / (img_max - img_min + 1e-6)
        
#         # 1. Lower threshold to catch faint outliers
#         binary_mask_gpu = (img > intensity_threshold).to(torch.uint8) * 255
#         binary_mask_np = binary_mask_gpu.cpu().numpy()
        
#         # 2. Dilate the mask to create a safety buffer and connect nearby blobs
#         # A circular kernel prevents creating boxy artifacts
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#         dilated_mask_np = cv2.morphologyEx(binary_mask_np, cv2.MORPH_DILATE, kernel, iterations=dilation_iters)
        
#         # 3. Send straight back to GPU as boolean
#         final_mask_gpu = torch.tensor(dilated_mask_np > 0, dtype=torch.bool, device=device)
#         # final_mask_gpu = torch.tensor(binary_mask_gpu, dtype=torch.bool, device=device)
#         masks.append(final_mask_gpu)
        
#     return masks

def extract_silhouettes(images, intensity_threshold=0.1):
    """
    Takes a list of float32 GPU tensors and returns a list of filled 
    boolean GPU tensors representing the swarm's convex hull.
    """
    # Assuming all images are on the same device (e.g., 'cuda:0')
    device = images[0].device
    hulls = []
    masks = []
    
    for img in images:
        # 1. Normalize directly on the GPU to leverage PyTorch speed
        img_min = img.min()
        img_max = img.max()
        img_norm = (img - img_min) / (img_max - img_min + 1e-6)
        
        # 2. Threshold on the GPU, format for OpenCV (uint8, 0-255)
        binary_mask_gpu = (img_norm > intensity_threshold).to(torch.uint8) * 255
        
        # Pull only the 2D byte array to the CPU (very fast)
        binary_mask_np = binary_mask_gpu.cpu().numpy()
        
        # 3. Morphological close to bridge small gaps between swarm particles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed_mask = cv2.morphologyEx(binary_mask_np, cv2.MORPH_CLOSE, kernel)

        masks.append(closed_mask)
        
        # 4. Find the contours of the swarm
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare an empty canvas for the filled mask
        final_mask_np = np.zeros_like(closed_mask)
        
        if contours:
            # Assume the swarm is the largest continuous blob in the image
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a Convex Hull (a "rubber band" wrapping the swarm)
            hull = cv2.convexHull(largest_contour)
            
            # Fill the inside of the hull with 1s
            cv2.drawContours(final_mask_np, [hull], -1, 1, thickness=cv2.FILLED)
        else:
            # Fallback if no contours are found (e.g., empty frame)
            final_mask_np = (closed_mask > 0).astype(np.uint8)
            
        # 5. Push the filled boolean mask back to the GPU for 3D carving
        final_mask_gpu = torch.tensor(final_mask_np, dtype=torch.bool, device=device)
        hulls.append(final_mask_gpu)
    
    return hulls
    # return hulls, masks

def hex_to_bgr(hex_code):
    """Converts a hex string (e.g., '#1f77b4') to a BGR tuple."""
    hex_code = hex_code.lstrip('#')
    # Convert hex to RGB integers
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    # Reverse to BGR for OpenCV
    return rgb[::-1]

def colorize_binary_array(binary_array, bgr_color):
    """
    binary_array: np.array (2D) containing 0s and 1s (or 255s)
    bgr_color: tuple, (Blue, Green, Red) e.g., (255, 0, 0) for Blue
    """
    # 1. Ensure the input is 2D (grayscale/binary)
    if len(binary_array.shape) != 2:
        raise ValueError("Input array must be 2D.")

    # 2. Create a white 3-channel canvas
    h, w = binary_array.shape
    colored_img = np.full((h, w, 3), 255, dtype=np.uint8)

    # 3. Identify "1s". We check for > 0 to handle both [0,1] and [0,255] formats
    mask = binary_array > 0

    # 4. Assign the user color to those coordinates
    colored_img[mask] = bgr_color

    return colored_img

def get_statistical_aabb(cameras, masks, sigma_multiplier=3.5):
    """
    Estimates the 3D bounding box by triangulating the 2D centers of mass 
    of the masks and scaling the volume using the 2D standard deviations.
    
    sigma_multiplier: How many standard deviations to include. 
                      3.0 covers ~99.7% of a Gaussian distribution. 
                      3.5 gives a little extra padding for the visual hull.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    ray_origins = []
    spreads_3d = []
    
    # 1. Calculate 2D statistics and generate 3D central rays
    for cam, mask in zip(cameras, masks):
        v, u = torch.where(mask > 0)
        if len(v) == 0:
            continue
            
        # 2D Mean (Center of mass) and Standard Deviation (Spread)
        u_mean, v_mean = u.float().mean().item(), v.float().mean().item()
        u_std, v_std = u.float().std().item(), v.float().std().item()
        
        # Camera intrinsics
        K = cam.state.intrinsics_params
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Ray direction in camera frame (unprojecting the mean pixel)
        dir_cam = np.array([(u_mean - cx) / fx, (v_mean - cy) / fy, 1.0])
        dir_cam /= np.linalg.norm(dir_cam)
        
        # Ray direction in world frame
        R_cam = cam.state.P_np[:, :3]
        T_cam = cam.state.P_np[:, 3]
        origin = cam.state.T_world  # True world location of camera
        
        dir_world = R_cam.T @ dir_cam
        dir_world /= np.linalg.norm(dir_world)
        
        ray_origins.append(origin)
        
        # Cache this camera's data for the radius calculation later
        spreads_3d.append({
            'u_std': u_std, 'v_std': v_std, 
            'fx': fx, 'fy': fy, 
            'R_cam': R_cam, 'T_cam': T_cam
        })
        
        # Accumulate matrices for least-squares ray intersection
        # We want the point P that minimizes distance to all rays
        I = np.eye(3)
        DDT = np.outer(dir_world, dir_world)
        A += (I - DDT)
        b += (I - DDT) @ origin

    # 2. Find the 3D Center (Closest point to all central rays)
    try:
        # Solve A * P = b
        center_3d = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Warning: Rays are parallel or poorly conditioned. Using mean of origins.")
        center_3d = np.mean(ray_origins, axis=0)
        
    # 3. Calculate 3D Extent (Radius) based on 2D Std Dev and Depth
    max_radius = 0.0
    for spread in spreads_3d:
        # Find the depth (Z) of our newly calculated 3D center relative to this camera
        cam_pt = spread['R_cam'] @ center_3d + spread['T_cam']
        z = cam_pt[2] 
        
        if z <= 0: 
            continue # Center is behind camera (shouldn't happen in a valid setup)
            
        # Translate 2D pixel standard deviation to 3D physical size at depth Z
        x_spread_3d = z * spread['u_std'] / spread['fx']
        y_spread_3d = z * spread['v_std'] / spread['fy']
        
        # Take the maximum spatial spread across all cameras as our base radius
        max_radius = max(max_radius, x_spread_3d, y_spread_3d)
            
    # 4. Construct the final AABB
    final_radius = max_radius * sigma_multiplier
    
    aabb_min = center_3d - final_radius
    aabb_max = center_3d + final_radius
    
    return np.stack([aabb_min, aabb_max], axis=1)

def reconstruct_visual_hull(cameras, images, scale=0.5, grid_max_size=32, grid_min_size=10, M=30, positions=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Get 2D Silhouettes
    # masks = extract_silhouettes(images, intensity_threshold=0.1)
    masks = extract_dilated_masks(images, intensity_threshold=0.05, dilation_iters=2)
    masks = [m.to(device) for m in masks]
    # user_hex = '#1f77b4'  # A nice muted blue
    # result = colorize_binary_array(masks[1].detach().cpu().numpy(), hex_to_bgr(user_hex))
    # cv2.imwrite('hex_colored_output.png', result)

    H, W = masks[0].shape
    
    # 2. Setup 3D Voxel Grid
    aabb = get_statistical_aabb(cameras, masks) # Ensure this returns [3, 2] array

    raw_grid_size = ((aabb[:, 1] - aabb[:, 0]) / (3 * scale)).astype(int)
    grid_size = np.clip(raw_grid_size, grid_min_size, grid_max_size)

    linspaces = [torch.linspace(aabb[j, 0].item(), aabb[j, 1].item(), grid_size[j], device=device) for j in range(3)]
    grids = torch.meshgrid(linspaces, indexing='ij')
    X, Y, Z = grids[0].ravel(), grids[1].ravel(), grids[2].ravel()
    
    ones = torch.ones_like(X)
    pnts_h = torch.stack([X, Y, Z, ones]) # Shape: (4, N_voxels)
    
    # Start assuming ALL voxels are part of the swarm
    valid_voxels_mask = torch.ones(pnts_h.shape[1], dtype=torch.bool, device=device)
    
    # 3. Voxel Carving (Intersection of viewing cones)
    for i, cam in enumerate(cameras):
        KP = torch.tensor(cam.state.K @ cam.state.P, device=device, dtype=torch.float32)
        proj = KP @ pnts_h
        
        # Perspective divide
        uv = proj[:2] / proj[2]
        z = proj[2]
        
        # Round to nearest pixel integer
        u = torch.round(uv[0]).long()
        v = torch.round(uv[1]).long()
        
        # Check which voxels project inside the image bounds and are in front of camera
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        
        # For voxels in bounds, check if they hit the 2D silhouette mask
        # Default to False (carved away) if out of bounds
        hits_mask = torch.zeros_like(valid_voxels_mask) 
        hits_mask[in_bounds] = masks[i][v[in_bounds], u[in_bounds]]
        
        # Logical AND: A voxel only survives if it's in the mask of EVERY camera
        valid_voxels_mask = valid_voxels_mask & hits_mask

        # peaks_pos_cpu = pnts_h[:3, valid_voxels_mask].cpu().numpy()
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # fig, ax = visualize_voxel_ellipsoid_mpl(peaks_pos_cpu.T, grid_size, aabb)
        # ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], s=1)
            
        # edges = [
        #     [0, 1], [1, 2], [2, 3], [3, 0],  # Near plane
        #     [4, 5], [5, 6], [6, 7], [7, 4],  # Far plane
        #     [0, 4], [1, 5], [2, 6], [3, 7]   # Side walls
        # ]
        
        # for j, cam_ in enumerate(cameras):
        #     vertices = cam_.state.get_world_frustum()
        #     for edge in edges:
        #         if j == i:
        #             ax.plot(
        #                 vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], 
        #                 color='orange', alpha=0.6, linewidth=1
        #             )
        #         else:
        #             ax.plot(
        #                 vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], 
        #                 color='royalblue', alpha=0.6, linewidth=1
        #             )
            
        # # Plot the camera optical center
        # center = cam.state.camera_center
        # cam_pt = ax.scatter(*center, c='royalblue', marker='s', s=30)
        
        # move_figure(fig, 100, 100)
        # ax.set_aspect('equal', 'box')
        # plt.show()

    # Get the final surviving 3D points
    occupied_points = pnts_h[:3, valid_voxels_mask].cpu()
    
    # 4. Calculate Volume
    num_surviving_voxels = valid_voxels_mask.sum().item()
    voxel_sizes_np = (aabb[:, 1] - aabb[:, 0]) / grid_size
    volume_per_voxel = np.prod(voxel_sizes_np)
    total_volume = num_surviving_voxels * volume_per_voxel
    
    # return occupied_points, total_volume
    f_peaks_pos = farthest_point_sampling(occupied_points, M)

    return aabb, grid_size, occupied_points, f_peaks_pos

def farthest_point_sampling(points, M):
    """
    Extracts M evenly spaced points from a point cloud.
    
    Args:
        points: PyTorch tensor of shape (3, N) or (N, 3)
        M: integer, number of points to extract
    Returns:
        sampled_points: PyTorch tensor of sampled points
    """
    # Handle shape so we are working with (N, 3) internally
    if points.shape[0] == 3:
        points = points.T

    N = points.shape[0]
    
    # Edge case: If the swarm is empty or has fewer than M points
    if N == 0:
        return points
    if N <= M:
        return points # Return all of them if M is larger than available points

    device = points.device
    selected_indices = torch.zeros(M, dtype=torch.long, device=device)
    
    # Initialize an array to keep track of the shortest distance from 
    # each point to the set of selected points. Start with infinity.
    distances = torch.ones(N, device=device) * 1e10
    
    # Randomly pick the first point
    farthest_idx = torch.randint(0, N, (1,), dtype=torch.long, device=device).item()
    
    for i in range(M):
        selected_indices[i] = farthest_idx
        
        # Get the coordinates of the newly selected point
        centroid = points[farthest_idx, :].view(1, 3)
        
        # Calculate squared Euclidean distance from all points to this new centroid
        # (Using squared distance saves us from computing expensive square roots)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        
        # Update the minimum distances for all points
        distances = torch.min(distances, dist)
        
        # The next selected point is the one with the maximum minimum distance
        farthest_idx = torch.argmax(distances, dim=-1).item()
        
    sampled_points = points[selected_indices]
    
    # Return in the (M, 3) shape
    return sampled_points

def get_voxel_peaks(cameras: list[Camera], images, scale=0.5, peak_threshold=0.3, grid_max_size=32, M=30, debug=False, positions=None):
    aabb = get_aabb(cameras)

    grid_size = np.clip(((aabb[:, 1] - aabb[:, 0]) / (3*scale)).astype(int), 
                        0, grid_max_size) # assume a maximum of 32^3 grid
    
    linspaces = [torch.linspace(aabb[j, 0].item(), aabb[j, 1].item(), grid_size[j], device='cuda') for j in range(3)]
    grids = torch.meshgrid(linspaces, indexing='ij')
    X, Y, Z = grids[0].ravel(), grids[1].ravel(), grids[2].ravel()
    ones = torch.ones_like(X)
    pnts_h = torch.stack([X, Y, Z, ones])

    KP_left_torch = cameras[0].state.K @ cameras[0].state.P
    KP_right_torch = cameras[1].state.K @ cameras[1].state.P

    proj = KP_left_torch @ pnts_h
    uv = proj[:2] / proj[2]

    proj2 = KP_right_torch @ pnts_h
    uv2 = proj2[:2] / proj2[2]

    h = images[0].shape[0]
    w = images[0].shape[1]
    mask = (uv[0] >= 0) & (uv[0] < w-1) & (uv[1] >= 0) & (uv[1] < h-1) & \
            (uv2[0] >= 0) & (uv2[0] < w-1) & (uv2[1] >= 0) & (uv2[1] < h-1)

    uv = uv[:, mask]
    uv2 = uv2[:, mask]
    valid_pnts = pnts_h[:3, mask]

    uv = uv.int()
    uv2 = uv2.int()

    vals_0 = images[0][uv[1], uv[0]]
    vals_1 = images[1][uv2[1], uv2[0]]
    voxel_intensity = vals_0 * vals_1

    valid_mask = (voxel_intensity >= peak_threshold)

    # post-processing
    peaks_voxel_intensity = voxel_intensity[valid_mask]
    peaks_pos = valid_pnts[:, valid_mask]
    f_peaks_pos, f_voxel_intensity = nms_3d_ultra_fast(peaks_pos.T, peaks_voxel_intensity, M, 6*scale)
    return aabb, grid_size, peaks_pos, peaks_voxel_intensity, f_peaks_pos, f_voxel_intensity
    
def nms_3d_ultra_fast(pos: torch.Tensor, intensity: torch.Tensor, M: int, radius: float):
    if pos.shape[0] == 0:
        return pos, intensity

    # 1. Sort upfront
    sorted_indices = torch.argsort(intensity, descending=True)
    pos = pos[sorted_indices]
    intensity = intensity[sorted_indices]

    # 2. Distance Matrix [N, N]
    # Use squared distance to avoid the square root (faster)
    r2 = radius ** 2
    # Manual squared distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # Often faster than cdist for small N
    dist_sq = torch.cdist(pos, pos) ** 2
    
    # 3. Adjacency Mask: True if points are too close
    adj = dist_sq < r2
    
    # 4. Greedy Selection (Optimized Loop)
    n = pos.shape[0]
    keep = torch.ones(n, dtype=torch.bool, device=pos.device)
    
    # Since N is tiny (150), we can use a bitmask approach or 
    # a highly efficient loop. Python's 'for' is slow, but we 
    # minimize operations inside it.
    adj_cpu = adj.cpu() # Small N? Moving to CPU can actually avoid kernel launch lag
    keep_cpu = torch.ones(n, dtype=torch.bool)
    
    for i in range(n):
        if keep_cpu[i]:
            # Mask out all subsequent points within radius
            keep_cpu[i+1:] &= ~adj_cpu[i, i+1:]
            
    # 5. Final Selection
    keep_indices = torch.where(keep_cpu)[0][:M]
    device_indices = keep_indices.to(pos.device)
    
    return pos[device_indices], intensity[device_indices]