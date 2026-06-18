import sys
import os
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import numpy as np
import torch
import torch.nn.functional as F
from dfr.mode_finding import find_scale_interval, mode_counting_pbc, mode_counting_modified_pbc, mode_counting, mode_counting_modified
from dfr.mode_finding import analytic_solution
from dfr.gaussian_mixture_reduction import GMR
from scipy.optimize import curve_fit

def analytic_solution_limit(x, d=2):
    if d == 2:
        xi = 1/(4*np.sqrt(3)*np.pi)
    else:
        if d == 3:
            xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
    return xi / x**d

# def analytic_solution_new_power_law(x, N=1, d=2):
#     A = 1
#     B = N
#     if d == 2:
#         k = 2 + 3.5269/N**0.2904 # no pbc
#         x0 = 0.36140247/N**0.53077842
#     else:
#         if d == 3:
#             k = 3 + 5.3439/N**0.2260 # no pbc
#             x0 = 0.3782/N**0.3613 # no pbc
#     return (B - A)/(1 + (x/x0)**k) + A

def analytic_solution_simple(x, N=1000, d=2, A=1):
    B = N
    if d == 2:
        xi = 1/(4*np.sqrt(3)*np.pi)
    else:
        if d == 3:
            xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
    x0 = (xi / N) ** (1 / d)
    return A + (B - A)/(1 + (x/x0)**d)

def analytic_solution_new(x, N=1000, d=2, A=1, pbc=False):
    A = 1
    B = N
    if d == 2:
        xi = 1/(4*np.sqrt(3)*np.pi)
        if pbc:
            k = 2 + 4.950595955353048 * N**-0.3515463947893045 # pbc
            x0 = 0.35597367503024324 * N**-0.5334539230881985
        else:
            k = 2 + 4.703183622216619 * N**-0.337789657437471 # no pbc
            x0 = 0.3991016137653485 * N**-0.5472965521349495
    else:
        if d == 3:
            xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
            if pbc:
                k = 3 + 6.985280925198838 * N**-0.3269168879106131 # pbc
                x0 = 0.3360391539543154 * N**-0.3551798732086752
            else:
                k = 3 + 6.939639899500436 * N**-0.2627997835686439 # no pbc
                x0 = 0.4028242833700554 * N**-0.3702108300180316 # no pbc
    # x0 = (2*xi / (N+1)) ** (1 / d)
    return A + (B - A)/(1 + (x/x0)**k)

def power_law(x, A, k):
    return A * x ** -k

def power_2pl(x, k, x0, A=1, D=0):
    return (A-D)/(1+(x/x0)**k) + D

def power_3pl(x, k, x0, gamma, A=1, D=0):
    return (A-D)/(1+(x/x0)**k)**gamma + D

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'tkagg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'wxagg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def verify_power_law_for_infinite_N():
    FORCE_UPDATE = True

    np.random.seed(19573)

    def generate_grf_2d(L, grid_num, rho_bar, R):
        # 1. Setup Grid and Frequencies (Using rfft for the last axis)
        dx = L / grid_num
        kx = 2 * np.pi * np.fft.fftfreq(grid_num, d=dx)
        ky = 2 * np.pi * np.fft.rfftfreq(grid_num, d=dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K_sq = KX**2 + KY**2

        # 2. Power Spectrum
        Pk = rho_bar * np.exp(-K_sq * R**2)
        Pk[0, 0] = 0  # Crucial: zero out the DC component

        # 3. Generate Complex Noise
        shape = K_sq.shape
        noise = (np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape)) / np.sqrt(2)
        
        # 4. Scale by Power Spectrum and normalize for 2D Volume
        # The (N / L) factor properly scales the continuous integral to discrete sum
        field_k = noise * np.sqrt(Pk) * (grid_num / L)
        
        # 5. Transform to Real Space
        # irfft2 handles Hermitian symmetry and returns an (N, N) real array
        field_fluctuations = np.fft.irfft2(field_k, s=(grid_num, grid_num))
        
        return rho_bar + field_fluctuations

    def generate_grf_3d(L, grid_num, rho_bar, R):
        # 1. Setup Grid and Frequencies
        dx = L / grid_num
        kx = 2 * np.pi * np.fft.fftfreq(grid_num, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(grid_num, d=dx)
        kz = 2 * np.pi * np.fft.rfftfreq(grid_num, d=dx)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K_mag_sq = KX**2 + KY**2 + KZ**2

        # 2. Define Power Spectrum
        Pk = rho_bar * np.exp(-K_mag_sq * R**2)
        Pk[0, 0, 0] = 0  # Zero out the DC component
        
        # 3. Generate Complex Noise
        shape = K_mag_sq.shape
        noise = (np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape)) / np.sqrt(2)
        
        # 4. Scale by Power Spectrum and normalize for 3D Volume
        # The (N / L)**1.5 factor ensures Var(field) = (1/L^3) * Sum(P(k))
        field_k = noise * np.sqrt(Pk) * (grid_num / L)**1.5
        
        # 5. Transform to Real Space
        field_fluctuations = np.fft.irfftn(field_k, s=(grid_num, grid_num, grid_num))
        
        return rho_bar + field_fluctuations
    
    L = 1 # linear scale of the domain
    grid_num = 600 # discretization
    rho_bar = 2 # mean of the field

    num_exp = 20
    num_test_scale = 20
    s_start = 1e-2
    s_end = 1e-1

    paths = {
        "modes_2d": os.path.join(os.getcwd(), "data_scaling_law", f"modes_2d_grf.npy"),
        "modes_3d": os.path.join(os.getcwd(), "data_scaling_law", f"modes_3d_grf.npy")
    }

    if FORCE_UPDATE or not os.path.exists(paths["modes_2d"]):
        modes_all_2d = []
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        for _ in range(num_exp):
            modes_ = []
            for s in test_scales:
                density_field = generate_grf_2d(L, grid_num, rho_bar, s)
                density_field_torch = torch.from_numpy(density_field).cuda().float()

                padded = F.pad(density_field_torch, pad=(1, 1), mode='circular')
                # padded = F.pad(density_field_torch, pad=(1, 1), mode='constant', value=float('-inf'))
                max_pool = F.max_pool2d(padded.unsqueeze(0), kernel_size=3, stride=1, padding=0, return_indices=False).squeeze(0)
                center_slice = padded[1:-1, 1:-1]
                maxima = (center_slice == max_pool) & (center_slice > 1e-5)
                modes_.append(torch.nonzero(maxima, as_tuple=False).shape[0])
            modes_all_2d.append(modes_)
        modes_all_2d = np.array(modes_all_2d)
        np.save(paths["modes_2d"], modes_all_2d)
    else:
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        modes_all_2d = np.load(paths["modes_2d"])
    
    # Set standard publication style
    plt.rcParams.update({
        "font.family": "serif",  # Use serif fonts for academic feel
        "font.size": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    def plot_with_inset_ratio(scales, mean, std, theory, title):
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # 1. Main Plot
        ax.plot(scales, mean, label='Numerical Mean', color='#2c3e50', lw=2)
        ax.fill_between(scales, mean-std, mean+std, color='#bdc3c7', alpha=0.5)
        ax.plot(scales, theory, label='Theoretical', color='#e74c3c', linestyle='--')
        
        # Hierarchy/Styling
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Number of Modes', fontsize=12)
        ax.set_xlabel('Scale ($\sigma$)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(frameon=False)
        
        # 2. Inset Ratio Plot
        # width/height are relative to parent axes
        ax_inset = inset_axes(ax, width="40%", height="35%", loc='lower left', borderpad=3.5)
        
        error = (mean - theory) / theory * 100
        ax_inset.plot(scales, error, color='black', lw=1.5)
        ax_inset.fill_between(scales, (mean-std - theory) / theory * 100, (mean+std - theory) / theory * 100, color='black', 
                              alpha=0.2, label='_nolegend_')
        ax_inset.axhline(1, color='red', linestyle=':', lw=1)
        
        # Styling the inset: Keep labels small to not compete with main plot
        ax_inset.set_title('Error (%)', fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        ax_inset.set_xscale('log')
        
        plt.tight_layout()
        return fig

    fig = plot_with_inset_ratio(test_scales, 
                          np.mean(modes_all_2d, axis=0), np.std(modes_all_2d, axis=0), 
                          1/(4*np.sqrt(3)*np.pi)*test_scales**-2, '2D Periodic GRF')
    move_figure(fig, 2800, 100)
    plt.savefig("figs/grf_modes_2d.png", bbox_inches='tight', dpi=300)
    # plt.savefig("figs/grf_modes_2d_no_pbc.png", bbox_inches='tight', dpi=300)

    grid_num = 300

    if FORCE_UPDATE or not os.path.exists(paths["modes_3d"]):
        modes_all_3d = []
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        for _ in range(num_exp):
            modes_ = []
            for s in test_scales:
                density_field = generate_grf_3d(L, grid_num, rho_bar, s)
                density_field_torch = torch.from_numpy(density_field).cuda().float()

                padded = F.pad(density_field_torch, pad=(1, 1, 1, 1), mode='circular')
                # padded = F.pad(density_field_torch, pad=(1, 1, 1, 1), mode='constant', value=float('-inf'))
                max_pool = F.max_pool3d(padded.unsqueeze(0), kernel_size=3, stride=1, padding=0, return_indices=False).squeeze(0)
                center_slice = padded[1:-1, 1:-1, 1:-1]
                maxima = (center_slice == max_pool) & (center_slice > 1e-8)
                modes_.append(torch.nonzero(maxima, as_tuple=False).shape[0])
            modes_all_3d.append(modes_)
        modes_all_3d = np.array(modes_all_3d)
        np.save(paths["modes_3d"], modes_all_3d)
    else:
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        modes_all_3d = np.load(paths["modes_3d"])

    fig = plot_with_inset_ratio(test_scales, 
                        np.mean(modes_all_3d, axis=0), np.std(modes_all_3d, axis=0), 
                        (29/6/np.sqrt(6) - 1)/8/np.pi**2*test_scales**-3, '3D Periodic GRF')
    move_figure(fig, 2800, 100)
    plt.savefig("figs/grf_modes_3d.png", bbox_inches='tight', dpi=300)
    # plt.savefig("figs/grf_modes_3d_no_pbc.png", bbox_inches='tight', dpi=300)

    plt.show()

def compute_scaling_law(force_update, dim=2, pbc=False, num_trials=5):
    # 1. Resolve naming and function references
    suffix = f"{dim}d" + ("_pbc" if pbc else "") + f"_trials{num_trials}"
    data_dir = os.path.join(os.getcwd(), "data_scaling_law")
    os.makedirs(data_dir, exist_ok=True)
    
    paths = {k: os.path.join(data_dir, f"{k}_{suffix}.{'npz' if k=='point_set' else 'npy'}") 
             for k in ["range", "point_set", "modes"]}
    
    mc_func = mode_counting_pbc if pbc else mode_counting
    mc_mod_func = mode_counting_modified_pbc if pbc else mode_counting_modified

    # 2. Setup Parameters
    num_test_rho = 40
    num_test_scale = 40
    test_rho = np.logspace(2, 4, num_test_rho)
    test_N = (test_rho).astype(int)

    # 3. Data Processing
    # Generate/Load points: Shape (num_trials, num_test_rho)
    pnt_set = np.empty((num_trials, num_test_rho), dtype=object)
    if force_update or not os.path.exists(paths["point_set"]):
        # Dictionary approach to save multiple arrays in one npz
        pnt_dict = {}
        for t in range(num_trials):
            for i in range(num_test_rho):
                pnt_dict[f't{t}_i{i}'] = np.random.uniform(0, 1, size=(test_N[i], dim))
                pnt_set[t, i] = pnt_dict[f't{t}_i{i}']
        np.savez(paths["point_set"], **pnt_dict)
    else:
        loaded = np.load(paths["point_set"])
        # Reconstruct structured access
        for t in range(num_trials):
            for i in range(num_test_rho):
                pnt_set[t, i] = loaded[f't{t}_i{i}']

    scale_range, all_modes, params = compute_scaling_law_(paths, num_test_scale, pnt_set, mc_func, mc_mod_func, force_update=False)

    return test_N, scale_range, all_modes, params

def compute_scaling_law_(paths, num_test_scale, pnt_set, mc_func, mc_mod_func, force_update=False):
    num_trials = len(pnt_set)
    num_scenarios = len(pnt_set[0])

    # Process Scales: Reuse range from the first trial for all others to save time
    if force_update or not os.path.exists(paths["range"]):
        # Only compute for the first trial (or average them if needed, but here we reuse)
        scale_range = []
        for i in tqdm(range(num_scenarios), desc=f"Processing Scale Range (Ref Trial)"):
            pos_gpu = torch.from_numpy(pnt_set[0][i]).cuda().float()
            f = lambda s: mc_func(pos_gpu, pos_gpu.clone(), s, max_iter=1000, tol=1e-3)
            s_start, s_end = find_scale_interval(f, pos_gpu.shape[0], s_initial_guess=15)
            scale_range.append([s_start, s_end])
        scale_range = np.array(scale_range)
        np.save(paths["range"], scale_range)
    else:
        scale_range = np.load(paths["range"])

    # Compute Modes: Now iterating over trials
    if force_update or not os.path.exists(paths["modes"]):
        all_modes = np.zeros((num_trials, num_scenarios, num_test_scale))
        
        for t in range(num_trials):
            for i in tqdm(range(num_scenarios), desc=f"Trial {t} | Computing #mode"):
                pos_gpu = torch.from_numpy(pnt_set[t][i]).cuda().float()
                s_start, s_end = scale_range[i]
                test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
                
                modes_pos = None
                for idx, s in enumerate(test_scales):
                    curr_pos = modes_pos.clone() if modes_pos is not None else pos_gpu.clone()
                    mode_num, tmp = mc_mod_func(pos_gpu, curr_pos, s, max_iter=2500, tol=5e-5)
                    modes_pos = torch.from_numpy(tmp).cuda().float()
                    all_modes[t, i, idx] = mode_num
        np.save(paths["modes"], all_modes)
    else:
        all_modes = np.load(paths["modes"])

    test_N = [i.shape[0] for i in pnt_set[0]]
    params = np.zeros((num_scenarios, 2)) 
    for i in range(num_scenarios):
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

        x_data = []
        y_data = []
        for j in range(num_trials):
            begin_idx = np.argmax(all_modes[j, i, :] <= 0.9 * test_N[i])
            
            x_data = np.hstack((x_data, test_scales[begin_idx:]))
            y_data = np.hstack((y_data, all_modes[j, i, begin_idx:]))
        
        # x_data = np.tile(test_scales, num_trials)
        # y_data = all_modes[:, i, :].flatten()
        
        try:
            popt, _ = curve_fit(
                lambda x, k, x0: power_2pl(x, k, x0, A=test_N[i], D=1), 
                x_data, 
                y_data, 
                p0=(2, 0),
                # Using y_data as sigma maintains the Poisson-like weighting from your original code
                sigma=y_data, 
                absolute_sigma=True, 
                bounds=([0]*2, [np.inf]*2)
            )
        except RuntimeError:
            print(f"Warning: Curve fit failed for pooled density index {i}")
            popt = [np.nan, np.nan]
        params[i] = popt

        # try:
        #     popt, _ = curve_fit(
        #         lambda x, k, x0, gamma: power_3pl(x, k, x0, gamma, A=test_N[i], D=0), 
        #         x_data, 
        #         y_data, 
        #         p0=(2, 0, 1),
        #         # Using y_data as sigma maintains the Poisson-like weighting from your original code
        #         sigma=y_data, 
        #         absolute_sigma=True, 
        #         bounds=([0]*3, [np.inf]*3)
        #     )
        # except RuntimeError:
        #     print(f"Warning: Curve fit failed for pooled density index {i}")
        #     popt = [np.nan]*3
        # params[i] = popt

    return scale_range, all_modes, params

def verify_convergence_for_finite_N():
    FORCE_UPDATE = False
    np.random.seed(12345678)

    num_test_scale = 40

    test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=2, pbc=True)

    selected_idx = [0, 13, 26, 39]
    selected_idx_start = [10, 2, 5, 5]
    selected_idx_end = [-1, -1, -1, -1]

    plt.rcParams.update({
        "font.family": "serif",  # Use serif fonts for academic feel
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax_inset = inset_axes(ax, width="35%", height="30%", loc='lower left', borderpad=3.5)
    move_figure(fig, 2800, 100)

    # Generate a gradient of blues (or another sequential map)
    # colors = cm.Blues(np.linspace(0.2, 1.0, len(selected_idx)))
    # colors = cm.viridis(np.linspace(0.1, 0.9, len(selected_idx)))
    colors = ['#85C1E9', '#3498DB', '#2874A6', '#154360']

    xi = 1/(4*np.sqrt(3)*np.pi)
    d = 2

    for idx, i in enumerate(selected_idx):
        s_start, s_end = scale_range[i]
        scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

        scales = scales[selected_idx_start[idx]:selected_idx_end[idx]]
        mean = np.mean(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        std = np.std(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        
        # Plotting: Using index-based color from our gradient
        ax.plot(scales, mean, color=colors[idx], lw=1.5, 
                label=f'$N={test_N[i]}$')
        
        # Important: label='_nolegend_' hides the fill from the legend list
        ax.fill_between(scales, mean-std, mean+std, color=colors[idx], 
                        alpha=0.2, label='_nolegend_')
        
        ax_inset.plot(scales, (mean - xi * scales**-d) / (xi * scales**-d) * 100, color=colors[idx], lw=1.5)
        ax_inset.fill_between(scales, 
                              (mean-std - xi * scales**-d) / (xi * scales**-d) * 100, 
                              (mean+std - xi * scales**-d) / (xi * scales**-d) * 100, color=colors[idx], 
                              alpha=0.2, label='_nolegend_')
        
        # Styling the inset: Keep labels small to not compete with main plot
        ax_inset.set_title(f'Error (%)', fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        ax_inset.set_xscale('log')
        ax_inset.set_ylim([-20, 20])

    # 2. Theoretical Limit (Make it bold/distinct)
    # We plot this last (or use zorder) so it sits on top

    tmp_ = np.max([np.max(num_modes[:, selected_idx[i], selected_idx_start[i]:selected_idx_end[i]]) for i in range(len(selected_idx_end))])
    tmp = np.min([np.min(num_modes[:, selected_idx[i], selected_idx_start[i]:selected_idx_end[i]]) for i in range(len(selected_idx_end))])
    scale_for_top = (xi / tmp_) ** (1/d)
    scale_for_bottom = (xi / tmp) ** (1/d)
    scales_ = np.logspace(np.log10(scale_for_top), np.log10(scale_for_bottom), num_test_scale)
    ax.plot(scales_, xi * scales_**-d, color='#e74c3c', linestyle='--', 
            lw=1.5, label=r'$N \to \infty$')

    # 3. Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('#Modes')
    ax.set_xlabel('Scale ($L$)')
    ax.set_title('Convergence of #Modes to Periodic GRF in 2D', fontsize=14)

    # Clean legend: Frameon=False keeps it minimalist
    ax.legend(frameon=False, loc='best')
    ax.grid(True, which='major', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("figs/converge_to_GRF_2d.png", bbox_inches='tight', dpi=300)

    test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=3, pbc=True)

    selected_idx = [0, 13, 26, 39]
    selected_idx_start = [0, 2, 4, 2]
    selected_idx_end = [-1, -1, -1, -1]

    plt.rcParams.update({
        "font.family": "serif",  # Use serif fonts for academic feel
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax_inset = inset_axes(ax, width="35%", height="30%", loc='lower left', borderpad=3.5)
    move_figure(fig, 2800, 100)

    # Generate a gradient of blues (or another sequential map)
    # colors = cm.Blues(np.linspace(0.2, 1.0, len(selected_idx)))
    # colors = cm.viridis(np.linspace(0.1, 0.9, len(selected_idx)))
    colors = ['#85C1E9', '#3498DB', '#2874A6', '#154360']

    xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
    d = 3

    for idx, i in enumerate(selected_idx):
        s_start, s_end = scale_range[i]
        scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

        scales = scales[selected_idx_start[idx]:selected_idx_end[idx]]
        mean = np.mean(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        std = np.std(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        
        # Plotting: Using index-based color from our gradient
        ax.plot(scales, mean, color=colors[idx], lw=1.5, 
                label=f'$N={test_N[i]}$')
        
        # Important: label='_nolegend_' hides the fill from the legend list
        ax.fill_between(scales, mean-std, mean+std, color=colors[idx], 
                        alpha=0.2, label='_nolegend_')
        
        # ax.plot(scales, power_2pl(scales, *params[i], A=test_N[i], D=1))
        
        ax_inset.plot(scales, (mean - xi * scales**-d) / (xi * scales**-d) * 100, color=colors[idx], lw=1.5)
        ax_inset.fill_between(scales, 
                              (mean-std - xi * scales**-d) / (xi * scales**-d) * 100, 
                              (mean+std - xi * scales**-d) / (xi * scales**-d) * 100, color=colors[idx], 
                              alpha=0.2, label='_nolegend_')
        
        # Styling the inset: Keep labels small to not compete with main plot
        ax_inset.set_title(f'Error (%)', fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        ax_inset.set_xscale('log')
        ax_inset.set_ylim([-30, 30])

    # 2. Theoretical Limit (Make it bold/distinct)
    # We plot this last (or use zorder) so it sits on top

    tmp_ = np.max([np.max(num_modes[:, selected_idx[i], selected_idx_start[i]:selected_idx_end[i]]) for i in range(len(selected_idx_end))])
    tmp = np.min([np.min(num_modes[:, selected_idx[i], selected_idx_start[i]:selected_idx_end[i]]) for i in range(len(selected_idx_end))])
    
    scale_for_top = (xi / tmp_) ** (1/d)
    scale_for_bottom = (xi / tmp) ** (1/d)
    scales_ = np.logspace(np.log10(scale_for_top), np.log10(scale_for_bottom), num_test_scale)
    ax.plot(scales_, xi * scales_**-d, color='#e74c3c', linestyle='--', 
            lw=1.5, label=r'$N \to \infty$')

    # 3. Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('#Modes')
    ax.set_xlabel('Scale ($L$)')
    ax.set_title('Convergence of #Modes to Periodic GRF in 3D', fontsize=14)

    # Clean legend: Frameon=False keeps it minimalist
    ax.legend(frameon=False, loc='best')
    ax.grid(True, which='major', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("figs/converge_to_GRF_3d.png", bbox_inches='tight', dpi=300)

    plt.show()


    test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=2, pbc=True)
    test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=3, pbc=True)

def finding_power_law_for_finite_N():
    FORCE_UPDATE = False

    np.random.seed(12345678)

    num_test_rho = 40
    num_test_scale = 40

    # nrows, ncols = 5, 8
    # fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    # move_figure(fig, 2700, 100)
    # fig.suptitle('2D pbc')
    # for i in range(num_test_rho):
    #     ax = axes.flat[i]
    #     s_start, s_end = scale_range[i]
    #     test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
    #     # for t in range(num_modes.shape[0]):
    #     #     ax.scatter(test_scales, num_modes[t][i], s=1)
    #     ax.plot(test_scales, power_2pl(test_scales, *params[i], A=test_N[i], D=1), '-', label='fitted')
    #     # ax.plot(test_scales, 1/(4*np.sqrt(3)*np.pi)*test_scales**-2, '--', label='infinite')
    #     # ax.plot(test_scales, analytic_solution_simple(test_scales, N=test_N[i], d=2, V=1), '--', label='simple')
        
    #     # ax.plot(test_scales, (29*np.sqrt(6)/288 - 1/8)/np.pi**2*test_scales**-3, '--', label='infinite')

    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    # plt.legend()

    # ----------------------------------------------
    # Finding D
    # error_all for D = 0 is 1827.57
    # error_all for D = 1 is 1378.33
    # scenarios = [
    #     {'dim': 2, 'pbc': False},
    #     {'dim': 3, 'pbc': False},
    #     {'dim': 2, 'pbc': True},
    #     {'dim': 3, 'pbc': True},
    # ]
    # error_all = 0
    # for sc in scenarios:
    #     test_N, scale_range, num_modes, params = compute_scaling_law(False, dim=sc['dim'], pbc=sc['pbc'])
        
    #     error_list = []
    #     for i in range(len(test_N)):
    #         s_start, s_end = scale_range[i]
    #         test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

    #         fit = power_2pl(test_scales, *params[i], A=test_N[i], D=1)

    #         error = np.abs(num_modes[i] - fit) / fit * 100
    #         error_list.append(np.mean(error).item())
    #     error_all += np.array(error_list).sum()
    # print(error_all)

    # ----------------------------------------------
    # Simple 4PL
    # test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=2, pbc=False)

    # selected_idx = [0, 13, 26, 39]
    # selected_idx_start = [10, 2, 5, 5]
    # selected_idx_end = [-1, -1, -1, -1]

    # plt.rcParams.update({
    #     "font.family": "serif",  # Use serif fonts for academic feel
    #     "font.size": 12,
    #     "axes.labelsize": 12,
    #     "legend.fontsize": 10,
    #     "xtick.direction": "in",
    #     "ytick.direction": "in",
    #     "axes.grid": True,
    #     "grid.alpha": 0.3,
    # })

    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax_inset = inset_axes(ax, width="35%", height="30%", loc='lower left', borderpad=3.5)
    # move_figure(fig, 2800, 100)

    # # Generate a gradient of blues (or another sequential map)
    # # colors = cm.Blues(np.linspace(0.2, 1.0, len(selected_idx)))
    # # colors = cm.viridis(np.linspace(0.1, 0.9, len(selected_idx)))
    # colors = ['#85C1E9', '#3498DB', '#2874A6', '#154360']
    # colors2 = cm.Oranges(np.linspace(0.3, 0.9, len(selected_idx)))

    # for idx, i in enumerate(selected_idx):
    #     s_start, s_end = scale_range[i]
    #     scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

    #     scales = scales[selected_idx_start[idx]:selected_idx_end[idx]]
    #     mean = np.mean(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
    #     std = np.std(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        
    #     # Plotting: Using index-based color from our gradient
    #     ax.plot(scales, mean, color=colors[idx], lw=1.5, 
    #             label=f'$N={test_N[i]}$')
        
    #     # predicted = analytic_solution_simple(scales, N=test_N[i], d=2, V=1)
    #     predicted = power_3pl(scales, *params[i], A=test_N[i], D=0)
    #     ax.plot(scales, predicted, color=colors2[idx], label=f'$N={test_N[i]}$ predict')
        
    #     # Important: label='_nolegend_' hides the fill from the legend list
    #     ax.fill_between(scales, mean-std, mean+std, color=colors[idx], 
    #                     alpha=0.2, label='_nolegend_')
        
    #     ax_inset.plot(scales, (mean - predicted) / predicted * 100, color=colors[idx], lw=1.5)
    #     ax_inset.fill_between(scales, 
    #                           (mean-std - predicted) / predicted * 100, 
    #                           (mean+std - predicted) / predicted * 100, color=colors[idx], 
    #                           alpha=0.2, label='_nolegend_')
        
    #     # Styling the inset: Keep labels small to not compete with main plot
    #     ax_inset.set_title(f'Error (%)', fontsize=10)
    #     ax_inset.tick_params(axis='both', which='major', labelsize=8)
    #     ax_inset.set_xscale('log')
    #     ax_inset.set_ylim([-40, 60])
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylabel('#Modes')
    # ax.set_xlabel('Scale ($L$)')
    # ax.set_title('#Modes in 2D', fontsize=14)
    # ax.legend()

    # test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=3, pbc=False)

    # fig2, ax2 = plt.subplots(figsize=(7, 5))
    # ax2_inset = inset_axes(ax2, width="35%", height="30%", loc='lower left', borderpad=3.5)
    # move_figure(fig2, 2800, 100)

    # for idx, i in enumerate(selected_idx):
    #     s_start, s_end = scale_range[i]
    #     scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

    #     scales = scales[selected_idx_start[idx]:selected_idx_end[idx]]
    #     mean = np.mean(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
    #     std = np.std(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        
    #     # Plotting: Using index-based color from our gradient
    #     ax2.plot(scales, mean, color=colors[idx], lw=1.5, 
    #             label=f'$N={test_N[i]}$')
        
    #     predicted = analytic_solution_simple(scales, N=test_N[i], d=3, V=1)
    #     ax2.plot(scales, predicted, color=colors2[idx], label=f'$N={test_N[i]}$ predict')
        
    #     # Important: label='_nolegend_' hides the fill from the legend list
    #     ax2.fill_between(scales, mean-std, mean+std, color=colors[idx], 
    #                     alpha=0.2, label='_nolegend_')
        
    #     ax2_inset.plot(scales, (mean - predicted) / predicted * 100, color=colors[idx], lw=1.5)
    #     ax2_inset.fill_between(scales, 
    #                           (mean-std - predicted) / predicted * 100, 
    #                           (mean+std - predicted) / predicted * 100, color=colors[idx], 
    #                           alpha=0.2, label='_nolegend_')
        
    #     # Styling the inset: Keep labels small to not compete with main plot
    #     ax2_inset.set_title(f'Error (%)', fontsize=10)
    #     ax2_inset.tick_params(axis='both', which='major', labelsize=8)
    #     ax2_inset.set_xscale('log')
    #     ax2_inset.set_ylim([-40, 80])
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_ylabel('#Modes')
    # ax2.set_xlabel('Scale ($L$)')
    # ax2.set_title('#Modes in 3D', fontsize=14)
    # ax2.legend()

    # ----------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 9,
        "legend.frameon": False
    })

    def power_law_offset(x, dim, b, c):
        return dim + b * (x**-c)

    def power_law_simple(x, b, c):
        return b * (x**-c)

    # Create a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    move_figure(fig, 2800, 100)

    # Styling dictionaries for consistency
    style_pbc = {'color': '#2c3e50', 'marker': 'o', 'label_prefix': 'PBC'}
    style_nopbc = {'color': '#e67e22', 'marker': '^', 'label_prefix': 'Non-PBC'}

    dimensions = [2, 3]
    pbcs = [True, False]

    for col_idx, d in enumerate(dimensions):
        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]
        
        ax_top.set_title(f'{d}D', pad=10)
        ax_top.set_ylabel('k')
        ax_bottom.set_ylabel(r'$\sigma_0$')
        
        for pbc in pbcs:
            style = style_pbc if pbc else style_nopbc
            
            # Load data (assuming FORCE_UPDATE is defined in your environment)
            test_N, scale_range, _, params = compute_scaling_law(FORCE_UPDATE, dim=d, pbc=pbc)
            
            # --- 1. TOP ROW: Offset Parameter ---
            y_off = params[:, 0]
            popt_off, _ = curve_fit(lambda x, b, c: power_law_offset(x, d, b, c), 
                                    test_N, y_off, p0=[3.5, 0.3], bounds=(0, np.inf))
            print(f'Fit k: ${d} + {popt_off[0]} *N^{{-{popt_off[1]}}}$ for pbc={pbc}, d={d}')
            
            ax_top.plot(test_N, power_law_offset(test_N, d, *popt_off), color=style['color'], lw=1.5,
                        label=rf"{style['label_prefix']} Fit: ${d} + {popt_off[0]:.1f} *N^{{-{popt_off[1]:.2f}}}$")
            ax_top.scatter(test_N, y_off, color=style['color'], alpha=0.4, s=20, marker=style['marker'])
            
            # --- 2. BOTTOM ROW: Scaling Parameter ---
            y_scale = params[:, 1]
            popt_s, _ = curve_fit(power_law_simple, test_N, y_scale, p0=[3.5, 0.5], bounds=(0, np.inf))
            print(f'Fit x0: ${popt_s[0]} *N^{{-{popt_s[1]}}} for pbc={pbc}, d={d}')
            
            ax_bottom.plot(test_N, power_law_simple(test_N, *popt_s), color=style['color'], lw=1.5,
                        label=rf"{style['label_prefix']} Fit: ${popt_s[0]:.1f} *N^{{-{popt_s[1]:.2f}}}$")
            
            # Add Theoretical Prediction ONLY once per column (we attach it to the on-PBC loop)
            if pbc is False:
                if d == 2:
                    theory = (2/(4*np.sqrt(3)*np.pi) / test_N)**(1/2)
                else:
                    xi = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
                    theory = (2*xi / (test_N+1))**(1/3)
                    
                ax_bottom.plot(test_N, theory, color='#e74c3c', linestyle='--', lw=2, zorder=10,
                            label=r'GRF limit ($N \to {\infty}$)')

            ax_bottom.scatter(test_N, y_scale, color=style['color'], alpha=0.4, s=20, marker=style['marker'])

        for ax in axes.flat:
            ax.set_xscale('log')
            ax.grid(True, which='major', linestyle=':', alpha=0.6)
            ax.legend(loc='best')

            ax.set_xlabel(r'$N$')

        for ax in axes[0, :]: # Selects all plots in the top row
            ax.tick_params(labelbottom=True)

    plt.tight_layout()
    plt.savefig("figs/k_and_x0_fit.png", bbox_inches='tight', dpi=300)


    test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=2, pbc=False)
    selected_idx = [0, 13, 26, 39]
    selected_idx_start = [10, 2, 5, 5]
    selected_idx_end = [-1, -1, -1, -1]
    plt.rcParams.update({
        "font.family": "serif",  # Use serif fonts for academic feel
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    ax_inset = inset_axes(ax, width="35%", height="30%", loc='lower left', borderpad=3.5)
    move_figure(fig, 2800, 100)

    # Generate a gradient of blues (or another sequential map)
    colors2 = cm.Oranges(np.linspace(0.2, 1.0, len(selected_idx)))
    # colors = cm.viridis(np.linspace(0.1, 0.9, len(selected_idx)))
    colors = ['#85C1E9', '#3498DB', '#2874A6', '#154360']

    xi = 1/(4*np.sqrt(3)*np.pi)
    d = 2

    for idx, i in enumerate(selected_idx):
        s_start, s_end = scale_range[i]
        scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

        scales = scales[selected_idx_start[idx]:selected_idx_end[idx]]

        mean = np.mean(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        std = np.std(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        
        # Plotting: Using index-based color from our gradient
        ax.plot(scales, mean, color=colors[idx], lw=1.5, 
                label=f'$N={test_N[i]}$')
        ax.fill_between(scales, mean-std, mean+std, color=colors[idx], 
                        alpha=0.2, label='_nolegend_')

        predicted = analytic_solution_new(scales, N=test_N[i], d=2, pbc=False)
        
        # Plotting: Using index-based color from our gradient
        ax.plot(scales, predicted, color=colors2[idx], lw=1.5, 
                label=f'$N={test_N[i]}$')
        
        # ax.plot(scales, analytic_solution_simple(scales, N=test_N[i], d=2), lw=1.5)
        
        ax_inset.plot(scales, (predicted - mean) / (mean) * 100, color=colors[idx], lw=1.5)
        ax_inset.fill_between(scales, 
                              (predicted-std - mean) / (mean) * 100, 
                              (predicted+std - mean) / (mean) * 100, color=colors[idx], 
                              alpha=0.2, label='_nolegend_')
        
        # Styling the inset: Keep labels small to not compete with main plot
        ax_inset.set_title(f'Error (%)', fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        ax_inset.set_xscale('log')
        ax_inset.set_ylim([-20, 20])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('4PL fitness 2D')

    test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=3, pbc=False)
    selected_idx = [0, 13, 26, 39]
    selected_idx_start = [0, 2, 4, 2]
    selected_idx_end = [-1, -1, -1, -1]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax_inset = inset_axes(ax, width="35%", height="30%", loc='lower left', borderpad=3.5)
    move_figure(fig, 2800, 100)

    # Generate a gradient of blues (or another sequential map)
    colors2 = cm.Oranges(np.linspace(0.2, 1.0, len(selected_idx)))
    # colors = cm.viridis(np.linspace(0.1, 0.9, len(selected_idx)))
    colors = ['#85C1E9', '#3498DB', '#2874A6', '#154360']

    xi = 1/(4*np.sqrt(3)*np.pi)
    d = 3

    for idx, i in enumerate(selected_idx):
        s_start, s_end = scale_range[i]
        scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

        scales = scales[selected_idx_start[idx]:selected_idx_end[idx]]

        mean = np.mean(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        std = np.std(num_modes[:, i, selected_idx_start[idx]:selected_idx_end[idx]], axis=0)
        
        # Plotting: Using index-based color from our gradient
        ax.plot(scales, mean, color=colors[idx], lw=1.5, 
                label=f'$N={test_N[i]}$')
        ax.fill_between(scales, mean-std, mean+std, color=colors[idx], 
                        alpha=0.2, label='_nolegend_')

        predicted = analytic_solution_new(scales, N=test_N[i], d=3, pbc=False)
        
        # Plotting: Using index-based color from our gradient
        ax.plot(scales, predicted, color=colors2[idx], lw=1.5, 
                label=f'$N={test_N[i]}$')
        
        # ax.plot(scales, analytic_solution_simple(scales, N=test_N[i], d=3), lw=1.5)
        
        ax_inset.plot(scales, (predicted - mean) / (mean) * 100, color=colors[idx], lw=1.5)
        ax_inset.fill_between(scales, 
                              (predicted-std - mean) / (mean) * 100, 
                              (predicted+std - mean) / (mean) * 100, color=colors[idx], 
                              alpha=0.2, label='_nolegend_')
        
        # Styling the inset: Keep labels small to not compete with main plot
        ax_inset.set_title(f'Error (%)', fontsize=10)
        ax_inset.tick_params(axis='both', which='major', labelsize=8)
        ax_inset.set_xscale('log')
        ax_inset.set_ylim([-20, 20])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('4PL fitness 3D')

    plt.show()

def discovery_of_dimensional_crossover():
    FORCE_UPDATE = False

    np.random.seed(35648216)

    num_test_rho = 40
    num_test_scale = 40
    num_trials = 5

    N = int(1e3)
    rho_low = 1e-3
    rho_high = 1
    num_test_rho = 40
    num_test_scale = 40
    test_rho = np.logspace(np.log10(rho_low), np.log10(rho_high), num_test_rho)
    width = 100
    length = 100
    A = width * length
    mean_2d_length = A ** (1/2)
    test_L = (N / test_rho) / A

    paths = {
        "range": os.path.join(os.getcwd(), "data_scaling_law", f"range_3d_constant_N_shrink_1d_{N}.npy"),
        "point_set": os.path.join(os.getcwd(), "data_scaling_law", f"point_set_3d_constant_N_shrink_1d_{N}.npz"),
        "modes": os.path.join(os.getcwd(), "data_scaling_law", f"modes_3d_constant_N_shrink_1d_{N}.npy"),
    }

    # 3. Data Processing
    # Generate/Load points: Shape (num_trials, num_test_rho)
    pnt_set = np.empty((num_trials, num_test_rho), dtype=object)
    if FORCE_UPDATE or not os.path.exists(paths["point_set"]):
        # Dictionary approach to save multiple arrays in one npz
        pnt_dict = {}
        for t in range(num_trials):
            for i in range(num_test_rho):
                pos = np.random.uniform(0, 1, size=(N, 3))
                pos[:, 0] = pos[:, 0] * width
                pos[:, 1] = pos[:, 1] * length
                pos[:, 2] = pos[:, 2] * test_L[i]
                pnt_dict[f't{t}_i{i}'] = pos
                pnt_set[t, i] = pnt_dict[f't{t}_i{i}']
        np.savez(paths["point_set"], **pnt_dict)
    else:
        loaded = np.load(paths["point_set"])
        # Reconstruct structured access
        for t in range(num_trials):
            for i in range(num_test_rho):
                pnt_set[t, i] = loaded[f't{t}_i{i}']
    
    # Process Scales: Reuse range from the first trial for all others to save time
    if FORCE_UPDATE or not os.path.exists(paths["range"]):
        # Only compute for the first trial (or average them if needed, but here we reuse)
        scale_range = []
        for i in tqdm(range(num_test_rho), desc=f"Processing Scale Range (Ref Trial)"):
            pos_gpu = torch.from_numpy(pnt_set[0, i]).cuda().float()
            f = lambda s: mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=1000, tol=1e-3)
            s_start, s_end = find_scale_interval(f, pos_gpu.shape[0], s_initial_guess=15)
            scale_range.append([s_start, s_end])
        scale_range = np.array(scale_range)
        np.save(paths["range"], scale_range)
    else:
        scale_range = np.load(paths["range"])
    
    # Compute Modes: Now iterating over trials
    if FORCE_UPDATE or not os.path.exists(paths["modes"]):
        all_modes = np.zeros((num_trials, num_test_rho, num_test_scale))
        
        for t in range(num_trials):
            for i in tqdm(range(num_test_rho), desc=f"Trial {t} | Computing #mode"):
                pos_gpu = torch.from_numpy(pnt_set[t, i]).cuda().float()
                s_start, s_end = scale_range[i]
                test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
                
                modes_pos = None
                for idx, s in enumerate(test_scales):
                    curr_pos = modes_pos.clone() if modes_pos is not None else pos_gpu.clone()
                    mode_num, tmp = mode_counting_modified(pos_gpu, curr_pos, s, max_iter=2000, tol=1e-4)
                    modes_pos = torch.from_numpy(tmp).cuda().float()
                    all_modes[t, i, idx] = mode_num
        np.save(paths["modes"], all_modes)
    else:
        all_modes = np.load(paths["modes"])
    
    params = np.zeros((num_test_rho, 2)) 
    for i in range(num_test_rho):
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        # Pool X data: repeat the scales array for each trial
        # Shape becomes (num_trials * num_test_scale,)
        x_data = np.tile(test_scales, num_trials)
        # Pool Y data: flatten the slice containing all trials for this specific density
        # Shape becomes (num_trials * num_test_scale,)
        y_data = all_modes[:, i, :].flatten()
        try:
            popt, _ = curve_fit(
                lambda x, k, x0: power_2pl(x, k, x0, A=N, D=1), 
                x_data, 
                y_data, 
                p0=(2, 0),
                # Using y_data as sigma maintains the Poisson-like weighting from your original code
                sigma=y_data, 
                absolute_sigma=True, 
                bounds=([0]*2, [np.inf]*2)
            )
        except RuntimeError:
            print(f"Warning: Curve fit failed for pooled density index {i}")
            popt = [np.nan, np.nan]
        params[i] = popt    

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    move_figure(fig, 2700, 100)

    xi_2 = 1/(4*np.sqrt(3)*np.pi)
    xi_3 = (29*np.sqrt(6)/288 - 1/8)/np.pi**2

    # Academic styling for 3D
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelpad": 15,  # 3D labels need more space
        "axes.titlesize": 18,
        "legend.fontsize": 12
    })

    # Remove the gray panes for a cleaner "modern" academic look
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # 1. Plot Numerical Curves with a Gradient
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    import matplotlib.patches as mpatches

    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D

    class HandlerFanningFill(HandlerBase):
        def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height,
                        fontsize, trans):
            
            # 1. Get the color from the dummy handle we pass to it
            color = orig_handle.get_color()
            
            # 2. Draw the solid mean line through the middle
            y_center = height / 2
            line = Line2D([0, width], [y_center, y_center], color=color, lw=2)
            
            # 3. Draw the expanding fill (a polygon acting as a proxy for fill_between)
            # Starts at a point on the left, expands to the full height on the right
            verts = [
                (0, y_center),           # Left point (pinched)
                (width, height * 1.0),   # Right top (expanded)
                (width, height * 0.0)    # Right bottom (expanded)
            ]
            fill = Polygon(verts, facecolor=color, alpha=0.2, edgecolor='none')
            
            # 4. Apply the legend's transform so it renders in the right place
            line.set_transform(trans)
            fill.set_transform(trans)
            
            # Return fill first so the line renders on top of it
            return [fill, line]

    for i in range(20):
        s_start, s_end = scale_range[i]
        test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
        mean = np.mean(all_modes[:, i, :], axis=0)
        std = np.std(all_modes[:, i, :], axis=0)
        
        y_val = i 
        y_line = np.full_like(test_scales, y_val)

        x_data = np.log10(test_scales)
        low_bound = np.log10(mean - std)
        high_bound = np.log10(mean + std)

        # 1. Create the vertices for the polygon
        # We trace the 'high' line forward, then the 'low' line backward to close the loop
        # Combine x and high_bound, then x and low_bound (reversed)
        # The 'y' value in your 3D plot is actually 'y_line' (the index i)
        xs = np.concatenate([x_data, x_data[::-1]])
        zs = np.concatenate([high_bound, low_bound[::-1]])
        
        # We create a list of (x, z) pairs
        poly_verts = list(zip(xs, zs))
        
        # 2. Create the PolyCollection
        # zs=[i] sets the position on the Y-axis (the depth)
        poly = PolyCollection([poly_verts], facecolors='#2c3e50', alpha=0.2)
        
        # 3. Add to the 3D plot
        # zdir='y' tells matplotlib that the 'zs=[i]' parameter refers to the Y-axis
        ax.add_collection3d(poly, zs=[i], zdir='y')
        
        # Keep your original line plot
        ax.plot(x_data, y_line, np.log10(mean), color='#2c3e50', lw=2, alpha=0.9, zorder=5)

    # 2. Surfaces: Use 'antialiased=True' and very low alpha for "ghost" reference
    x_range = np.linspace(0.1, 1, 100)
    y_range = np.linspace(2, 20, 100)
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate your full Z matrices as normal
    Z2D = np.log10(xi_2*A) - 2*X
    Z3D = np.log10(xi_3*A) + np.log10(N/A/rho_low) + \
        np.log10(rho_low/rho_high) / (num_test_rho - 1) * Y - 3 * X

    # Create masks for where each plane is on top
    mask_2d_top = Z2D > Z3D
    mask_3d_top = Z3D >= Z2D

    # Split Z2D into top and bottom
    Z2D_top = np.where(mask_2d_top, Z2D, np.nan)
    Z2D_hidden = np.where(~mask_2d_top, Z2D, np.nan)

    # Split Z3D into top and bottom
    Z3D_top = np.where(mask_3d_top, Z3D, np.nan)
    Z3D_hidden = np.where(~mask_3d_top, Z3D, np.nan)

    # Plot the dominant (top) regions with higher opacity and shading
    ax.plot_surface(X, Y, Z2D_top, color='royalblue', alpha=0.25, 
                    edgecolor='none', shade=False)

    ax.plot_surface(X, Y, Z3D_top, color='darkorange', alpha=0.25, 
                    edgecolor='none', shade=False)

    # Optional: Plot the hidden (sub-dominant) regions as very faint ghost surfaces
    ax.plot_surface(X, Y, Z2D_hidden, color='royalblue', alpha=0.15, 
                    edgecolor='none', shade=False)

    ax.plot_surface(X, Y, Z3D_hidden, color='darkorange', alpha=0.15, 
                    edgecolor='none', shade=False)

    # 4. Transition Curve: Make it stand out
    intersection_scale = xi_3/xi_2 * np.array([test_L[5], test_L[18]])
    ax.plot(np.log10(intersection_scale), [5, 18], 
            np.log10(xi_2*A/intersection_scale**2), 
            linestyle='--', color='#e74c3c', lw=3, label='Transition Curve', zorder=10)

    # --- Axis Formatting ---
    y_ticks = np.linspace(0, 22.5, 9)
    logh = np.log10(N/A/rho_low) + np.log10(rho_low/rho_high) / (num_test_rho - 1) * y_ticks

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{x:.1f}" for x in logh])

    ax.set_zlim([-0.4, 3.0])

    ax.set_xlabel(r'$\log_{10}(\sigma)$', fontsize=12)
    ax.set_ylabel(r'$\log_{10}(h)$', fontsize=12)
    ax.set_zlabel(r'$\log_{10}(m)$', fontsize=12)

    # Adjust the viewing angle to see the intersection clearly
    ax.view_init(elev=20, azim=-27)

    # Tighten layout and add a light grid
    ax.grid(True) # Default 3D grids are often messy, keeping it minimal

    # 1. Create a dummy line that holds your specific color
    numerical_proxy = Line2D([0], [0], color='#2c3e50')

    # 2. Rebuild your proxy list
    proxies = [
        numerical_proxy,
        mpatches.Patch(facecolor='royalblue', alpha=0.2),
        mpatches.Patch(facecolor='darkorange', alpha=0.2),
        Line2D([0], [0], color='#e74c3c', lw=3, linestyle='--')
    ]

    labels = ['Numerical (Mean ± Std)', 'GRF limit 2d', 'GRF limit 3d', 'Transition Curve']

    # 3. Call the legend and map the dummy line to your custom handler
    ax.legend(
        handles=proxies, 
        labels=labels,
        handler_map={numerical_proxy: HandlerFanningFill()}, # The magic happens here
        frameon=False, 
        loc=(0.05, 0.75), # Adjust based on your view angle
        handlelength=3.0  # Optional: gives the fan more horizontal space to expand
    )
    
    ax.set_title('Dimensional Crossover', y=1)

    ax.set_box_aspect((1, 1, 0.8))
    ax.set_position([0, 0, 1, 1])

    plt.savefig("figs/dimensional_crossover.png", dpi=300)
    # plt.show()

    import cv2

    def crop_by_percentage(image, crop_coords):
        """
        Crops an image based on percentages.
        
        Args:
            image: The input image (NumPy array).
            crop_coords: A tuple (left, right, bottom, top) as floats from 0.0 to 1.0.
                        e.g., (0.1, 0.9, 0.2, 0.8)
        """
        left_pct, right_pct, bottom_pct, top_pct = crop_coords
        
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Convert percentages to pixel indices
        # Note: 'top' and 'bottom' are relative to the y-axis (0 is top)
        x1 = int(width * left_pct)
        x2 = int(width * right_pct)
        
        # In image arrays, y increases as you go down.
        # We assume 'top' means closer to the top edge (smaller Y)
        # and 'bottom' means closer to the bottom edge (larger Y).
        y1 = int(height * top_pct)
        y2 = int(height * bottom_pct)
        
        # Crop using NumPy slicing: image[y1:y2, x1:x2]
        # We use min/max or specific order to ensure we don't get an empty slice
        cropped_img = image[y1:y2, x1:x2]
        
        return cropped_img

    def display_fixed_size(image, window_name="Cropped Image", width=800, height=600):
        # Create a window that can be resized
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Resize the window to your fixed dimensions
        cv2.resizeWindow(window_name, width, height)
        
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img = cv2.imread('figs/dimensional_crossover.png')
    cropped = crop_by_percentage(img, (0.15, 0.9, 0.95, 0))
    display_fixed_size(cropped)
    cv2.imwrite('figs/dimensional_crossover.png', cropped)

    # plt.rcParams.update({
    #     "font.family": "serif",  # Use serif fonts for academic feel
    #     "font.size": 14,
    #     "axes.labelsize": 12,
    #     "legend.fontsize": 10,
    #     "xtick.direction": "in",
    #     "ytick.direction": "in",
    #     "axes.grid": True,
    #     "grid.alpha": 0.3,
    # })

    # selected_idx = list(range(4, 17, 4))

    # colors = cm.gist_yarg(np.linspace(0.35, 0.7, len(selected_idx)))
    # colors2 = cm.Oranges(np.linspace(0.35, 0.7, len(selected_idx)))
    # fig, ax = plt.subplots(figsize=(7, 5))
    # move_figure(fig, 2800, 100)
    # for i, idx in enumerate(selected_idx):
    #     s_start, s_end = scale_range[idx]
    #     test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)

    #     mean = np.mean(all_modes[:, idx, :], axis=0)
    #     std = np.std(all_modes[:, i, :], axis=0)
    #     # ax.plot(test_scales, mean, color='#2c3e50', label=r'$\nu$' + f' = {test_L[idx]/A**0.5:.2f} Numerical', lw=2)
    #     ax.plot(test_scales, mean, color=colors[i], label=f'h = {test_L[idx]:.2f} Numerical', lw=2)
    #     ax.fill_between(test_scales, mean-std, mean+std, color='#bdc3c7', alpha=0.5)

    #     ax.plot(test_scales[:-1-4*i], analytic_solution_new(test_scales[:-1-4*i]/(test_L[idx]*A)**(1/3), N=1000, d=3, A=1, pbc=False), '--', color=colors2[i], 
    #             label=f'h = {test_L[idx]:.2f} 4PL')

    # s_start, s_end = scale_range[0]
    # test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
    # ax.plot(test_scales[10:], analytic_solution_limit(test_scales[10:]/A**(1/2), d=2), '--', color="tab:blue", label='GRF limit 2d', lw=2.5)
    # # ax.plot(test_scales[10:], analytic_solution_new(test_scales[10:]/A**(1/2), N=1000, d=2), '--', label='2d')

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylabel('Number of Modes', fontsize=12)
    # ax.set_xlabel('Scale ($\sigma$)', fontsize=12)
    # ax.legend()

    # plt.savefig("figs/dimensional_crossover_2.png", dpi=300)
    # plt.show()

def veriry_dimensional_crossover():
    from dfr.simulation_config import SimulationConfig
    from dfr.dataset_io import DatasetFactory
    FORCE_UPDATE = False

    num_test_scale = 40

    # 1. Parameter extraction and Logging Setup
    name = 'starling'

    scenario_path = os.path.join(os.getcwd(), *["scenarios", name])
    config_path = os.path.join(scenario_path, "config.yaml")
    
    # 3. Load Dataset
    config = SimulationConfig(config_path) 
    factory = DatasetFactory()
    dataset = factory.get_dataset(config.data_file)

    time_step = 0
    positions = dataset.positions_at_time_step(time_step)

    pos_gpu = torch.from_numpy(positions).cuda().float()
    nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
    avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()

    N = positions.shape[0]

    paths = {
            "range": os.path.join(os.getcwd(), "data_scaling_law", f"{name}_t_{time_step}_scale_range.npy"),
            "modes": os.path.join(os.getcwd(), "data_scaling_law", f"{name}_t_{time_step}_modes.npy"),
        }

    # Processing Scale Range
    if FORCE_UPDATE or not os.path.exists(paths["range"]):
        f = lambda s: mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=avg_nn_dist / 1e3)
        s_start, s_end = find_scale_interval(f, pos_gpu.shape[0], s_initial_guess=30)
        scale_range = [s_start, s_end]
        # scale_range = [np.exp(np.log(s_start) - (np.log(s_end) - np.log(s_start)) * 0.1), 
        #                     s_end + 0.2 * (s_end - s_start)] 
        
        scale_range = np.array([scale_range])
        np.save(paths["range"], scale_range)
    else:
        scale_range = np.load(paths["range"])
    
    s_start, s_end = scale_range[0]
    test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
    # Computing #mode
    if FORCE_UPDATE or not os.path.exists(paths["modes"]):
        # modes = [mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=1e-4) for s in test_scales]
        num_modes = []
        modes_pos = None
        for s in tqdm(test_scales, desc=f"Processing scale"):
            if modes_pos is None:
                mode_num, tmp = mode_counting_modified(pos_gpu, pos_gpu.clone(), s, max_iter=3000, tol=avg_nn_dist / 1e5)
            else:
                mode_num, tmp = mode_counting_modified(pos_gpu, modes_pos.clone(), s, max_iter=3000, tol=avg_nn_dist / 1e5)
            modes_pos = torch.from_numpy(tmp).cuda().float()
            num_modes.append(mode_num)
        
        num_modes = np.array([num_modes])
        np.save(paths["modes"], num_modes)
    else:
        num_modes = np.load(paths["modes"])

    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    import open3d as o3d

    def extract_manifold(data, k=20):
        nbrs = NearestNeighbors(n_neighbors=k).fit(data)
        _, indices = nbrs.kneighbors(data)
        
        denoised_points = np.zeros_like(data)
        
        thicknesses = []
        dist_distribution = []
        normals_list = []
        for i, idx_list in enumerate(indices):
            neighbor_points = data[idx_list]
            pca = PCA(n_components=3)
            pca.fit(neighbor_points)
            
            # The first two components define the 2D plane
            # Projecting the central point onto this local plane
            mean = pca.mean_
            normal = pca.components_[2] # The direction of "thickness"
            
            # Project point onto the plane: P_new = P - ((P - mean) · normal) * normal
            vec = data[i] - mean
            dist_from_plane = np.dot(vec, normal)
            denoised_points[i] = data[i] - dist_from_plane * normal

            dist_distribution.append(dist_from_plane)

            # If the "thickness" is Gaussian noise: 2σ captures about 95% of the points. This is a solid "effective thickness" estimate.
            # If the "thickness" is a uniform slab: For a uniform distribution of height T, the variance is σ**2=T**2/12​. 
            # In that case, the true thickness would be T=\sqrt(12⋅λ3)​​≈3.46σ.
            thickness = 3 * np.sqrt(pca.explained_variance_[2])
            thicknesses.append(thickness)

            normals_list.append(normal)
            
        return denoised_points, np.array(thicknesses), np.array(dist_distribution), np.array(normals_list)

    def calculate_manifold_area(points):
        # 1. Convert numpy array to Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 2. Estimate Normals (Required for surface reconstruction)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # 3. Create a Mesh using Ball Pivoting
        # Note: radius should be slightly larger than the average distance between points
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2])
        )

        # volume = mesh.get_volume()

        # 4. Check if the mesh is watertight
        if not mesh.is_watertight():
            print("Warning: Mesh is not watertight. Volume might be inaccurate.")

        # 4. Calculate Area
        surface_area = mesh.get_surface_area()
        return surface_area, mesh

    denoised, thicknesses, dist_distribution, normals = extract_manifold(positions, k=150)
    area, manifold_mesh = calculate_manifold_area(denoised)
    # o3d.visualization.draw_geometries([manifold_mesh])
    print(f"Total Surface Area: {area:.4f}")

    # 1. Ensure the Open3D mesh has vertex normals calculated
    manifold_mesh.compute_vertex_normals()

    # 2. Extract the clean geometry from the Open3D mesh
    vertices = np.asarray(manifold_mesh.vertices)
    triangles = np.asarray(manifold_mesh.triangles)
    vertex_normals = np.asarray(manifold_mesh.vertex_normals)

    # 3. Determine the displacement (thickness)
    # For a clean visualization of the tube, using the mean thickness is easiest.
    # (If you need variable thickness, you would map your 'thicknesses' array to these 'vertices' via a KDTree)
    avg_thickness = np.mean(thicknesses)
    print(f"Effective thickness: {avg_thickness:.4f}")
    half_t = avg_thickness / 2.0
    xi_2 = 1/(4*np.sqrt(3)*np.pi)
    xi_3 = (29*np.sqrt(6)/288 - 1/8)/np.pi**2
    print(f"Relavent range of height: [{np.sqrt(xi_2**3/xi_3**2*area/N/0.5):.4f}, {np.sqrt(xi_2**3/xi_3**2*area/N/0.01):.4f}]")

    # 4. Calculate the upper and lower boundary vertices
    # We push the central vertices along their normal vectors
    upper_vertices = vertices + (vertex_normals * half_t)
    lower_vertices = vertices - (vertex_normals * half_t)

    # fig, ax = plt.subplots(figsize=(7, 5))
    # move_figure(fig, 2800, 100)
    # # plt.scatter(np.arange(len(dist_distribution)), dist_distribution)
    # plt.hist(dist_distribution, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # plt.show()
    # plt.show()

    import matplotlib.patches as mpatches
    from matplotlib import cm

    # --- Standard Academic Styling ---
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # 3D labels need extra space so they don't overlap the tick labels
        "axes.labelpad": 12 
    })

    # 1. Create the Figure with Dual Subplots
    fig = plt.figure(figsize=(16, 8))
    move_figure(fig, 2500, 100) # Slightly adjusted position for wider figure

    # fig.suptitle(f"Manifold Reconstruction (Surface Area: {area:.2f})", fontsize=16)

    # Define our professional color palette
    color_points = '#2c3e50'    # Deep Navy (muted)
    color_manifold = '#e67e22'  # Muted Orange (complementary to navy)
    color_boundary = '#bdc3c7'  # Soft Gray/Silver

    # Defined viewing angles from your request
    views = [
        {'elev': -20, 'azim': 80,  'roll': 180,  'title': 'Front View'},
        {'elev': -35, 'azim': -16, 'roll': -178, 'title': 'Side View'}
    ]

    for idx, v in enumerate(views):
        # Create the subplot (1 row, 2 columns, sharez=True is key!)
        ax = fig.add_subplot(1, 2, idx+1, projection='3d', sharez=(None if idx==0 else axes[0]))
        if idx == 0: axes = [ax] # Store the first axis for sharing later

        # Clean up the "boxy" 3D look by removing the gray panes
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # --- Plotting Elements ---

        # 1. Original Points (Scatter)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=color_points, s=4, alpha=0.5, zorder=1)

        # 2. Central Manifold ( plot_trisurf )
        # Using 'shade=False' to maintain a flat, clear color
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                        triangles=triangles, color=color_manifold, alpha=0.3, 
                        linewidth=0, antialiased=True, shade=False, zorder=5)

        # 3. Boundaries ( plot_trisurf - very transparent )
        # These create the "tube" effect
        ax.plot_trisurf(upper_vertices[:, 0], upper_vertices[:, 1], upper_vertices[:, 2], 
                        triangles=triangles, color=color_boundary, alpha=0.2, 
                        linewidth=0, antialiased=True, shade=False, zorder=10)

        ax.plot_trisurf(lower_vertices[:, 0], lower_vertices[:, 1], lower_vertices[:, 2], 
                        triangles=triangles, color=color_boundary, alpha=0.2, 
                        linewidth=0, antialiased=True, shade=False, zorder=10)

        # --- Formatting each subplot ---
        # ax.set_title(v['title'], fontsize=14, pad=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=v['elev'], azim=v['azim'], roll=v['roll'])
        
        # Keep grid light and subtle
        ax.grid(True, which='major', linestyle=':', alpha=0.5)

    # 3. Create a unified legend for the whole figure
    point_proxy = mpatches.Patch(color=color_points, alpha=0.5, label='Original Data')
    manifold_patch = mpatches.Patch(color=color_manifold, alpha=0.3, label='Reconstructed Manifold')
    boundary_patch = mpatches.Patch(color=color_boundary, alpha=0.2, label='Thickness Boundary')

    fig.legend(handles=[point_proxy, manifold_patch, boundary_patch], 
            loc='upper center', ncol=3, frameon=False, fontsize=12, borderaxespad=2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make room for legend/title
    plt.savefig("figs/starling_dimensional_crossover.png", bbox_inches='tight', dpi=300)
    plt.show()

    # # def update(frame):
    # #     # Change the azimuth angle (frame represents the current step)
    # #     ax.view_init(elev=20, azim=frame)
    # #     return fig,

    # # from matplotlib.animation import FuncAnimation
    # # ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=50)
    # # ani.save('concave_hull_rotation.mp4', writer='ffmpeg', fps=30)
    # plt.show()

    plt.rcParams.update({
        "font.family": "serif",  # Use serif fonts for academic feel
        "font.size": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    move_figure(fig, 2800, 100)

    ax.plot(test_scales, num_modes[0], label='Numerical Mean', color='#2c3e50', lw=2)

    ax.plot(test_scales, analytic_solution_new(test_scales/area**(1/2), N=N, d=2), '--', label='2d')
    ax.plot(test_scales, analytic_solution_new(test_scales/(area*avg_thickness)**(1/3), N=N, d=3), '--', label='3d')

    xi_2 = 1/(4*np.sqrt(3)*np.pi)
    xi_3 = (29*np.sqrt(6)/288 - 1/8)/np.pi**2

    ax.scatter(xi_3/xi_2*avg_thickness, xi_2**3/xi_3**2/avg_thickness**2 * area, color='#e74c3c', label='Transition Point')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('Number of Modes', fontsize=12)
    ax.set_xlabel('Scale ($\sigma$)', fontsize=12)
    ax.legend()

    plt.savefig("figs/starling_dimensional_crossover_2.png", bbox_inches='tight', dpi=300)
    plt.show()

# def exploring_quasi_crystal():
#     FORCE_UPDATE = True

#     num_test_scale = 40

#     # 1. Parameter extraction and Logging Setup
#     name = 'quasi_crystal'

#     import numpy as np
#     import matplotlib.pyplot as plt

#     def generate_quasicrystal(size=500, n_fold=5, scale=50):
#         # Create a coordinate grid
#         x = np.linspace(-scale, scale, size)
#         y = np.linspace(-scale, scale, size)
#         X, Y = np.meshgrid(x, y)
        
#         # Initialize the pattern
#         pattern = np.zeros((size, size))
        
#         # Sum the plane waves
#         for i in range(n_fold):
#             # Calculate the angle for this wave vector
#             angle = np.pi * i / n_fold
#             # Project coordinates onto the wave vector direction
#             # This is effectively the dot product k_j * r
#             projection = X * np.cos(angle) + Y * np.sin(angle)
#             # Add the cosine wave to the pattern
#             pattern += np.cos(projection)
            
#         return pattern
    
#     def generate_crystal_positions(nx=20, ny=20, lattice_type='hexagonal', a=1.0):
#         """
#         Generates discrete atomic positions for standard 2D crystals.
#         a: lattice constant (spacing)
#         nx, ny: number of unit cells in x and y directions
#         """
#         # Define basis vectors for different lattice types
#         if lattice_type == 'square':
#             v1 = np.array([a, 0])
#             v2 = np.array([0, a])
#         elif lattice_type == 'hexagonal':
#             v1 = np.array([a, 0])
#             v2 = np.array([a/2, a * np.sqrt(3)/2])
#         elif lattice_type == 'rectangular':
#             v1 = np.array([a, 0])
#             v2 = np.array([0, a * 1.5]) # Aspect ratio of 1.5
#         else:
#             raise ValueError("Lattice type not supported.")

#         # Create a grid of indices
#         i = np.arange(nx)
#         j = np.arange(ny)
#         I, J = np.meshgrid(i, j)
        
#         # Calculate positions: R = i*v1 + j*v2
#         x_coords = I * v1[0] + J * v2[0]
#         y_coords = I * v1[1] + J * v2[1]
        
#         return x_coords.flatten(), y_coords.flatten()

#     from scipy.ndimage import maximum_filter
#     def get_peak_coordinates(pattern, neighborhood_size=5, threshold=1.0):
#         # 1. Apply a maximum filter to find local maxima
#         local_max = maximum_filter(pattern, size=neighborhood_size) == pattern
        
#         # 2. Create a mask to ignore low-intensity noise/background
#         # (Adjust threshold based on your specific pattern intensity)
#         mask = (pattern > threshold)
        
#         # 3. Combine them to get coordinates
#         peaks = local_max & mask
#         y_coords, x_coords = np.where(peaks)
        
#         return x_coords, y_coords

#     # Generate the pattern (from previous step)
#     # qc = generate_quasicrystal(size=800, n_fold=12, scale=100)
#     # x_p, y_p = get_peak_coordinates(qc, neighborhood_size=10, threshold=1.5)
#     x_p, y_p = generate_crystal_positions(nx=30, ny=30, lattice_type='square')

#     positions = np.vstack((x_p, y_p)).T
#     positions = positions / 30


#     pos_gpu = torch.from_numpy(positions).cuda().float()
#     nn_dist = torch.cdist(pos_gpu, pos_gpu) + torch.eye(pos_gpu.shape[0], device='cuda') * 1e10
#     avg_nn_dist = torch.median(torch.min(nn_dist, dim=1).values).item()

#     N = positions.shape[0]

#     paths = {
#             "range": os.path.join(os.getcwd(), "data_scaling_law", f"{name}_t_scale_range.npy"),
#             "modes": os.path.join(os.getcwd(), "data_scaling_law", f"{name}_t_modes.npy"),
#         }

#     # Processing Scale Range
#     if FORCE_UPDATE or not os.path.exists(paths["range"]):
#         f = lambda s: mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=avg_nn_dist / 1e3)
#         s_start, s_end = find_scale_interval(f, pos_gpu.shape[0], s_initial_guess=5)
#         scale_range = [s_start, s_end]
#         # scale_range = [np.exp(np.log(s_start) - (np.log(s_end) - np.log(s_start)) * 0.1), 
#         #                     s_end + 0.2 * (s_end - s_start)] 
        
#         scale_range = np.array([scale_range])
#         np.save(paths["range"], scale_range)
#     else:
#         scale_range = np.load(paths["range"])
    
#     s_start, s_end = scale_range[0]
#     test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
#     # Computing #mode
#     if FORCE_UPDATE or not os.path.exists(paths["modes"]):
#         # modes = [mode_counting(pos_gpu, pos_gpu.clone(), s, max_iter=2000, tol=1e-4) for s in test_scales]
#         num_modes = []
#         modes_pos = None
#         for s in tqdm(test_scales, desc=f"Processing scale"):
#             if modes_pos is None:
#                 mode_num, tmp = mode_counting_modified(pos_gpu, pos_gpu.clone(), s, max_iter=3000, tol=avg_nn_dist / 1e5)
#             else:
#                 mode_num, tmp = mode_counting_modified(pos_gpu, modes_pos.clone(), s, max_iter=3000, tol=avg_nn_dist / 1e5)
#             modes_pos = torch.from_numpy(tmp).cuda().float()
#             num_modes.append(mode_num)
        
#         num_modes = np.array([num_modes])
#         np.save(paths["modes"], num_modes)
#     else:
#         num_modes = np.load(paths["modes"])


#     plt.rcParams.update({
#         "font.family": "serif",  # Use serif fonts for academic feel
#         "font.size": 14,
#         "axes.labelsize": 12,
#         "legend.fontsize": 10,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "axes.grid": True,
#         "grid.alpha": 0.3,
#     })

#     fig, ax = plt.subplots(figsize=(7, 5))
#     move_figure(fig, 2800, 100)

#     ax.plot(test_scales, num_modes[0], label='Numerical Mean', color='#2c3e50', lw=2)

#     ax.plot(test_scales, analytic_solution_new(test_scales/1**(1/2), N=N, d=2), '--', label='2d')

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_ylabel('Number of Modes', fontsize=12)
#     ax.set_xlabel('Scale ($\sigma$)', fontsize=12)
#     ax.legend()

#     plt.savefig("figs/starling_dimensional_crossover_2.png", bbox_inches='tight', dpi=300)
#     plt.show()

def calculate_gmm_covariance(means: torch.Tensor, weights: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """
    Calculates the global covariance matrix of an isotropic GMM.
    
    Args:
        means: Tensor of shape [N, 3] containing the centers.
        weights: Tensor of shape [N, 1] containing the mixture weights.
        radius: Tensor of shape [N, 1] containing the isotropic standard deviation.
        
    Returns:
        global_cov: Tensor of shape [3, 3] representing the global covariance matrix.
    """
    # 1. Ensure weights sum to 1
    w = weights / weights.sum()
    
    # 2. Calculate the global mean
    # Broadcasting w [N, 1] over means [N, 3]
    global_mean = (w * means).sum(dim=0, keepdim=True) # Shape: [1, 3]
    
    # 3. Calculate the covariance from the spread of the component means
    # Center the means
    centered_means = means - global_mean # Shape: [N, 3]
    
    # Compute: Sum of w_i * (mu_i - mu) * (mu_i - mu)^T
    # Using matrix multiplication for efficiency: (3, N) @ (N, 3) -> (3, 3)
    cov_from_means = (w * centered_means).T @ centered_means
    
    # 4. Calculate the covariance from the individual isotropic components
    # Assuming 'radius' represents standard deviation, so variance is radius^2.
    # (If 'radius' is already variance, just use 'radius' instead of 'radius ** 2')
    variances = radius ** 2
    
    # Weighted sum of variances
    global_var_scalar = (w * variances).sum()
    
    # Construct the isotropic covariance matrix part
    cov_from_components = global_var_scalar * torch.eye(3, device=means.device, dtype=means.dtype)
    
    # 5. Total Covariance
    global_cov = cov_from_means + cov_from_components
    
    return global_cov

def finding_effective_volume():
    FORCE_UPDATE = False

    np.random.seed(12345678)

    num_test_rho = 40
    num_test_scale = 40
    num_trials = 5

    # nrows, ncols = 5, 8
    # fig, axes = plt.subplots(nrows, ncols, figsize=(20, 10))
    # move_figure(fig, 2700, 100)
    # fig.suptitle('2D pbc')
    # for i in range(num_test_rho):
    #     ax = axes.flat[i]
    #     s_start, s_end = scale_range[i]
    #     test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
    #     # for t in range(num_modes.shape[0]):
    #         # ax.scatter(test_scales, num_modes[t][i], s=1)
    #     # ax.plot(test_scales, power_2pl(test_scales, *params[i], A=test_N[i], D=1), '-', label='fitted')
    #     ax.plot(test_scales, 1/(4*np.sqrt(3)*np.pi)*test_scales**-2, '--', label='infinite')
    #     ax.plot(test_scales, analytic_solution_simple(test_scales, N=test_N[i], d=2, V=1), '--', label='simple')
        
    #     # ax.plot(test_scales, (29*np.sqrt(6)/288 - 1/8)/np.pi**2*test_scales**-3, '--', label='infinite')

    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    # plt.legend()

    # num_modes num_trials, num_test_rho, num_test_scale
    test_N, scale_range, num_modes, params = compute_scaling_law(FORCE_UPDATE, dim=3, pbc=False)

    data_dir = os.path.join(os.getcwd(), "data_scaling_law")
    pnt_sets_ = np.load(os.path.join(data_dir, "point_set_3d_trials5.npz"))
    pnt_set = np.empty((num_trials, num_test_rho), dtype=object)
    for t in range(num_trials):
        for i in range(num_test_rho):
            pnt_set[t, i] = pnt_sets_[f't{t}_i{i}']

    idx_scale = -5
    blurred_volume = []
    exact_volume = []
    for t in range(3):
        for rho in range(10):
            print(t, rho)
            selected_pnt_set = pnt_set[t, rho]
            N = selected_pnt_set.shape[0]
            num_modes[t, rho]
            s_start, s_end = scale_range[rho]
            test_scales = np.logspace(np.log10(s_start), np.log10(s_end), num_test_scale)
            selected_scale = test_scales[idx_scale]

            r_means, r_weights, r_covs = GMR.runnalls_algorithm_simple_torch(
                means=torch.from_numpy(selected_pnt_set),
                radii=torch.full((N, 1), selected_scale, device='cuda', dtype=torch.float),
                weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
                L=20, DEVICE='cuda'
            )
            r_radius = torch.sqrt(r_covs[:, 0, 0]).reshape((-1, 1))
            # final_means, final_unnorm_weights, final_covs = GMR.optimize_ise_isotropic(
            #     orig_means=torch.from_numpy(selected_pnt_set),
            #     orig_covs=(torch.eye(3, device='cuda') * (selected_scale ** 2)).unsqueeze(0).expand(N, 3, 3),
            #     orig_weights=torch.full((N, 1), 1.0, device='cuda', dtype=torch.float),
            #     reduced_means=r_means,
            #     reduced_covs=r_covs,
            #     reduced_weights=r_weights,
            #     num_iterations=200,
            #     lr_mu_pct = 0.05,  # Percentage of data scale (e.g., 5%)
            #     lr_var = 0.05,     # Fixed LR for log-variances
            #     lr_weight = 0.05,  # Fixed LR for softmax logits
            #     DEVICE='cuda'
            # )
            # final_unnorm_weights = final_unnorm_weights.reshape((-1, 1))
            # final_radius = torch.sqrt(final_covs[:, 0, 0]).reshape((-1, 1))

            cov = calculate_gmm_covariance(r_means, r_weights.reshape((-1, 1)), r_radius) - selected_scale ** 2 * torch.eye(3, device='cuda', dtype=torch.float32)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=0.0)
            radii = 2 * torch.sqrt(eigenvalues)
            blurred_volume.append(((4/3) * np.pi * torch.prod(radii)).item())

    
            cov = torch.cov(torch.from_numpy(selected_pnt_set.astype(np.float32)).T)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=0.0)
            radii = 2 * torch.sqrt(eigenvalues)
            exact_volume.append(((4/3) * np.pi * torch.prod(radii)).item())
    
    blurred_volume = np.array(blurred_volume)
    exact_volume = np.array(exact_volume)

    print(np.abs(blurred_volume - exact_volume) / exact_volume)

if __name__ == "__main__":
    # verify_power_law_for_infinite_N()

    verify_convergence_for_finite_N()
    # finding_power_law_for_finite_N()

    # discovery_of_dimensional_crossover()
    # veriry_dimensional_crossover()

    # finding_effective_volume()