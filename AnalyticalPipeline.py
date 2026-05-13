import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects, center_of_mass
from scipy.signal import find_peaks, find_peaks_cwt
from skimage.transform import warp_polar
from pathlib import Path
from tqdm import tqdm
import os
from math import sqrt 
from scipy.signal import find_peaks
import torch 
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from mbhl.geometry import Geometry, Mesh, Circle, Rectangle, Square 
from mbhl.simulation import Stencil, Physics, System 
import torch.nn as nn 
from DatasetGeneration import Universal_Square_Lattice, Universal_Diamond_Lattice, Universal_Hexagonal_Lattice, Universal_Honeycomb_Lattice
from NanoLithoForwardModel import UNet, load_weights
from torch.utils.data import Dataset, DataLoader
import random

class FilteredDataset:
    """
    Wrapper that filters a dataset by N range.
    """
    def __init__(self, original_dataset, n_range, num_samples=None, seed=42):
        """
        Args:
            original_dataset: SimulationDataset_Backward instance
            n_range: tuple (min_n, max_n) e.g., (1, 30)
            samples_per_n: if None, take all samples; if int, take that many per N
            seed: random seed for reproducibility
        """

        self.original_dataset = original_dataset
        self.n_min, self.n_max = n_range
        
        # Find all indices with N in range
        self.indices = []
        self.n_to_indices = {}
        
        print(f"Filtering dataset for N in [{self.n_min}, {self.n_max}]...")
        count = 0 
        for idx in tqdm(range(len(original_dataset))):
            _, _, _, true_N, _ = original_dataset[idx]
            if self.n_min <= true_N <= self.n_max:
                self.indices.append(idx)
                count += 1 
                
                if count == num_samples: 
                    break
        
        print(f"Filtered to {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


class SimulationDataset_Backward(Dataset):
    def __init__(self, file_path, train=True, split=0.8):
        paths = list(Path(file_path).glob('*.npz'))
        random.seed(42)
        random.shuffle(paths)
        split_idx = int(len(paths) * split)
        self.paths = paths[:split_idx] if train else paths[split_idx:]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx], allow_pickle=True)
        stencil = np.nan_to_num(data['input_stencil'])
        target  = np.nan_to_num(data['target_deposition'])
        traj    = data['trajectory']
        point_count = len(traj)
        target_normalized = target / point_count

        stencil_tensor = torch.tensor(stencil).float().unsqueeze(0)
        target_tensor  = torch.tensor(target_normalized).float().unsqueeze(0)

        traj[:, 0] /= (2 * np.pi)
        traj[:, 1] /= 0.1

        # Helper function to safely convert to float
        def safe_float(val):
            if val is None or val == b'None' or str(val) == 'None':
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        # Helper function to safely convert to string
        def safe_str(val):
            if val is None or val == b'None':
                return None
            return str(val)

        # Load simulation parameters (with safe conversion)
        params = {
            'lattice': safe_str(data['lattice']),
            'shape': safe_str(data['shape']),
            'rotation': safe_float(data['rotation']) if 'rotation' in data else 0.0,
            'width': safe_float(data['width']) if 'width' in data else None,
            'radius': safe_float(data['radius']) if 'radius' in data else None,
            'height': safe_float(data['height']) if 'height' in data else None,
            'orientation': safe_str(data['orientation']) if 'orientation' in data else 'vertical',
            'diffusion': safe_float(data['diffusion']) if 'diffusion' in data else 15e-9,
            'RL_ratio': safe_float(data['RL_ratio']) if 'RL_ratio' in data else 1.0,
        }
        
        return (stencil_tensor, target_tensor, torch.tensor(traj).float(), point_count, params)




class BackwardLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mse    = nn.MSELoss()
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)

    def forward(self, recon, target):
        return 0.90 * self.mse(recon, target) + 0.10 * (1 - self.msssim(recon, target))

def get_centermost_center(centers, image_shape):
    h, w = image_shape
    cx_img, cy_img = w/2.0, h/2.0
    dist2 = (centers[:,0] - cx_img)**2 + (centers[:,1] - cy_img)**2
    idx = np.argmin(dist2)
    return centers[idx]

def find_aperture_centers(stencil, threshold=0.5):
    """stencil: 2D array (H,W), values 0/1 or 0/255. Returns list of (cx,cy)."""
    binary = stencil > threshold
    labeled, n = label(binary)
    centers = []
    for i in range(1, n+1):
        ys, xs = np.where(labeled == i)
        cx = np.mean(xs)
        cy = np.mean(ys)
        centers.append((cx, cy))
    #get centermost center: 
    center = get_centermost_center(centers = np.array(centers), image_shape = stencil.shape)
    return center


def get_shape_diagonal(shape_type, w=None, h=None, r=None):
    """
    Calculate the maximum distance from shape center to any point (the circumradius).
    This is half the diagonal for squares/rectangles, radius for circles.
    
    Returns:
    - diagonal: maximum distance from center to edge
    """
    if shape_type == 'circle':
        return r
    
    elif shape_type == 'square':
        # Half the diagonal of the square
        return (w * sqrt(2)) / 2
    
    elif shape_type == 'rectangle':
        # Half the diagonal of the rectangle
        return sqrt(w**2 + h**2) / 2
    
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")    

def get_min_distance(lattice_type, L):
    if lattice_type == 'square':
        return L

    elif lattice_type == 'hexagonal':  # triangular lattice
        return L

    elif lattice_type == 'honneycomb':
        return L / sqrt(3)

    elif lattice_type == 'diamond':
        return sqrt(3) * L / 4

    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

def ThetaSearch(target, lattice_type, center, L=500e-3, pixel_size=5e-3, N_bins=15):

    # 1. physical → physical
    d_min = get_min_distance(lattice_type=lattice_type, L=L)

    # 2. physical → pixels (IMPORTANT FIX)
    search_radius = d_min / pixel_size
    search_radius = search_radius / 2 
    cx, cy = center

    # 3. bounding box in pixel space
    #constructing "square" of search space (in pixels)
    x_min = int(np.floor(cx - search_radius))
    x_max = int(np.ceil(cx + search_radius))
    y_min = int(np.floor(cy - search_radius))
    y_max = int(np.ceil(cy + search_radius))

    #gives all pixel indices
    x = np.arange(x_min, x_max + 1)
    y = np.arange(y_min, y_max + 1)

    #returns 2 2D grids corresponding to x and y coordinates (still pixel)
    xx, yy = np.meshgrid(x, y)

    # 4. convert to polar coords
    dx = xx - cx
    dy = yy - cy

    r2 = dx**2 + dy**2
    mask = r2 <= search_radius**2

    dx = dx[mask]
    dy = dy[mask]

    theta = np.arctan2(dy, dx)
    theta = (theta + 2*np.pi) % (2*np.pi)   # [0, 2π]

    # clip indices to valid image bounds
    xx_i = np.clip(xx[mask], 0, target.shape[1] - 1).astype(int)
    yy_i = np.clip(yy[mask], 0, target.shape[0] - 1).astype(int)

    values = target[yy_i, xx_i]
    

    # 6. angular bins
    bins = np.linspace(0, 2*np.pi, N_bins + 1)

    hist, _ = np.histogram(
        theta,
        bins=bins,
        weights=values
    )

    # 7. normalize → density
    hist = hist / (hist.sum() + 1e-12)

    


    return hist, bins, (dx, dy, values, theta) 


def build_phi_distribution_per_theta(dx, dy, values, theta, N_theta = 15, N_r = 100, phi_max = 0.1, N_phi = 20, height = 1000):
    r = np.sqrt(dx**2 + dy**2) 

    theta_bins = np.linspace(0, 2*np.pi, N_theta + 1) 
    theta_idx = np.digitize(theta, theta_bins) - 1

    phi_bins = np.linspace(0, 0.1, N_phi + 1) 
    r_bins = np.linspace(0, r.max() + 1e-12, N_r + 1)

    #convert radii to phi and clip to remain physically consistent
    

    phi_dists = []

    for t in range(N_theta): 
        mask = (theta_idx == t) 
        
        if np.sum(mask) < 5: 
            phi_dists.append(np.ones(N_phi) / N_r)
            continue 

        r_t = r[mask]

        w_t = values[mask]

        phi_t = np.arctan(r_t / height) 
        phi_t = np.clip(phi_t, 0, phi_max)

        

        #basically building: for every theta, making a radial distribution where the weight is the intensity at that radius
        #and then we are going to append this to  multiple thetas
        hist, _ = np.histogram(phi_t, bins=phi_bins, weights = w_t)

        hist = np.clip(hist, 0, None) 

        #Normalize: 
        prob = hist / (hist.sum() + 1e-12)

        #CDF mapping: 
        cdf = np.cumsum(prob)
        cdf = cdf / (cdf[-1] + 1e-12)

        #append it to the phi-theta map 
        phi_dists.append(cdf) 


    return phi_dists

def sample_from_cdf(cdf, bins):
    """
    Invert a discrete CDF properly.
    """
    u = np.random.rand()
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 0, len(bins)-1)

    # sample uniformly inside bin (adds necessary randomness)
    if idx == 0:
        return bins[0]
    return np.random.uniform(bins[idx-1], bins[idx])


def GeneratePrior(theta_dists, phi_dists, N, N_t=15, N_phi=20, temperature=1.0):
    """
    theta_dists: p(theta)
    phi_dists: list of p(phi | theta)
    """

    trajectory = np.zeros((N, 2))

    theta_bins = np.linspace(0, 2*np.pi, N_t + 1)
    phi_bins = np.linspace(0, 0.1, N_phi + 1)

    # -------------------------
    # 1. sample theta indices
    # -------------------------
    theta_probs = np.array(theta_dists)
    theta_probs = theta_probs / (theta_probs.sum() + 1e-12)

    theta_probs = np.power(theta_probs, 1/temperature)
    theta_probs = theta_probs / theta_probs.sum()

    theta_idx_samples = np.random.choice(N_t, size=N, p=theta_probs)

    # -------------------------
    # 2. sample (theta, phi)
    # -------------------------
    for i in range(N):

        t_idx = theta_idx_samples[i]

        # θ = bin midpoint
        theta = 0.5 * (theta_bins[t_idx] + theta_bins[t_idx+1])

        trajectory[i, 0] = theta

        # φ distribution for this θ
        cdf = phi_dists[t_idx]

        # ensure valid
        if len(cdf) == 0:
            trajectory[i, 1] = np.random.uniform(0, 0.1)
            continue

        # sample φ via inverse CDF
        phi = sample_from_cdf(cdf, phi_bins)

        trajectory[i, 1] = phi

    return trajectory


def refine_trajectory(forward_model, stencil, target, init_traj, N, device,
                      lr=1e-3, max_steps=2000, target_loss=None,  # None = no early stop
                      patience=30, verbose=True):
    """
    Full trajectory refinement.
    If target_loss is None: runs to patience/max_steps (no early convergence)
    If target_loss is set: stops early if loss < target_loss
    """
    forward_model.eval()
    
    traj = init_traj.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([traj], lr=lr)
    best_loss = float('inf')
    best_traj = traj.detach().clone()
    no_improve = 0
    loss_fn = BackwardLoss(device=device)
    
    
    if verbose:
        print(f"  [Refine] N={N} | max_steps={max_steps} | target_loss={target_loss if target_loss else 'none'}")
    
    for step in range(max_steps):
        traj_c = traj.clamp(0.0, 1.0)

        
        recon = forward_model({
            'stencil': stencil,
            'traj': [traj_c[0]],
            'num_pts': torch.tensor([N], device=device)
        })
        loss = loss_fn(recon, target)
        
        # Early convergence check (only if target_loss is set)
        if target_loss is not None and loss.item() < target_loss:
            if verbose:
                print(f"  [Refine] Converged early at step {step} | loss={loss.item():.6f}")
            return traj_c.detach(), loss.item(), True
        
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_traj = traj.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            if verbose:
                print(f"  [Refine] Patience exhausted at step {step} | best_loss={best_loss:.6f}")
            return best_traj.clamp(0.0, 1.0).detach(), best_loss, False
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    if verbose:
        print(f"  [Refine] Max steps reached | best_loss={best_loss:.6f}")
    return best_traj.clamp(0.0, 1.0).detach(), best_loss, False


def generate_prior_trajectory(stencil_tensor, target_tensor, params, N_points,
                              N_theta=15, N_phi=20, N_r=100, phi_max=0.1, height=1000,
                              pixel_size=5e-3, L=500e-3, temperature=1.0):
    """
    Generate initial trajectory using analytical prior.
    Returns tensor of shape (1, N_points, 2) with normalized (theta, phi) in [0,1].
    """
    # Convert tensors to numpy
    stencil_np = stencil_tensor[0, 0].cpu().numpy()   # assume shape (1,1,H,W)
    target_np = target_tensor[0, 0].cpu().numpy()

    # 1. Find aperture centre from stencil
    center = find_aperture_centers(stencil_np, threshold=0.5)   # (cx, cy) in pixels

    # 2. Get angular distribution from target
    lattice_type = params['lattice']
    hist, bins, (dx, dy, values, theta) = ThetaSearch(
        target_np, lattice_type, center, L=L, pixel_size=pixel_size, N_bins=N_theta
    )

    # 3. Build conditional phi distributions
    phi_dists = build_phi_distribution_per_theta(
        dx, dy, values, theta,
        N_theta=N_theta, N_r=N_r, phi_max=phi_max, N_phi=N_phi, height=height
    )

    # 4. Sample trajectory (theta in [0,2π], phi in [0, phi_max])
    traj_phys = GeneratePrior(
        hist, phi_dists, N_points,
        N_t=N_theta, N_phi=N_phi, temperature=temperature
    )   # shape (N_points, 2)  columns: theta (rad), phi (rad)

    # 5. Normalize to [0,1] for refinement
    traj_norm = np.zeros_like(traj_phys)
    traj_norm[:, 0] = traj_phys[:, 0] / (2 * np.pi)   # theta norm
    traj_norm[:, 1] = traj_phys[:, 1] / phi_max       # phi norm

    # 6. Add batch dimension and convert to tensor
    traj_tensor = torch.tensor(traj_norm, dtype=torch.float32).unsqueeze(0)  # (1, N, 2)
    return traj_tensor

def reconstruct_stencil_from_params(params, L=500e-9, H=5e-6, default_h=5e-9):
    """
    Recreate the original Stencil object from saved parameters.
    """
    lattice_type = params['lattice']
    shape_type = params['shape']
    rotation = params['rotation']
    w = params['width']
    r = params['radius']
    h = params['height']
    orientation = params['orientation']
    
    # Select lattice function
    if lattice_type == 'hexagonal':
        lattice_func = Universal_Hexagonal_Lattice
    elif lattice_type == 'square':
        lattice_func = Universal_Square_Lattice
    elif lattice_type == 'diamond':
        lattice_func = Universal_Diamond_Lattice
    elif lattice_type == 'honneycomb':
        lattice_func = Universal_Honeycomb_Lattice
    else:
        raise ValueError(f"Unknown lattice: {lattice_type}")
    
    # Build arguments
    args = {'L': L, 'type': shape_type, 'rotation': rotation}
    if shape_type == 'circle':
        args['r'] = r
    elif shape_type == 'square':
        args['w'] = w
    else:  # rectangle
        args['w'] = w
        args['h'] = h
    
    if lattice_type in ['hexagonal', 'honneycomb']:
        args['orientation'] = orientation
    
    geometry = lattice_func(**args)
    stencil = Stencil(geometry, thickness=0, gap=H, h=default_h)
    return stencil

def run_ground_truth_simulation(stencil, trajectory_phys, diffusion, 
                                L=500e-9, H=5e-6, extra=1, point_count=None):
    """
    Run the original simulation code with given stencil and physical trajectory.
    Returns 256x256 deposition pattern normalized to match dataset format.
    """
    phys = Physics(trajectory_phys, diffusion=diffusion, drift=0)
    system = System(stencil=stencil, physics=phys)
    fold_to_bz = True
    F, M_padded, M_origin, recover_indices, extra = system._prepare_matrices(
        add_diffusion=True, fold_to_bz=fold_to_bz
    )
    system.simulate(method="fft", fold_to_bz=fold_to_bz)
    target_raw = system.results.tiled_mesh(extra_x=extra).array
    from skimage.transform import resize
    target_ready = resize(target_raw, (256, 256), order=1, anti_aliasing=True)
    
    if point_count is not None:
        target_ready = target_ready / point_count
    
    return target_ready

# ---------- Modified golden search ----------
def golden_search_with_full_refinement(forward_model, get_prior_fn, stencil, target,
                                        device, low, high, target_loss=None,
                                        max_iter=6, verbose=True):
    """
    Golden section search using analytical prior for initialisation.
    get_prior_fn: callable that takes an integer N and returns initial trajectory tensor.
    """
    phi = (1 + 5**0.5) / 2
    inv_phi = 1.0 / phi

    a = float(low)
    b = float(high)

    evaluated = {}

    def evaluate_full(Nc):
        Nc = int(round(Nc))
        if Nc in evaluated:
            return evaluated[Nc]

        if verbose:
            print(f"Evaluating N={Nc}...")

        # Generate initial trajectory using analytical prior
        init_traj = get_prior_fn(Nc)

        # Full refinement with early stopping
        refined_traj, final_loss, converged = refine_trajectory(
            forward_model, stencil, target, init_traj, Nc, device,
            target_loss=target_loss, patience=30, verbose=verbose
        )

        evaluated[Nc] = (final_loss, refined_traj, converged)
        return final_loss, refined_traj, converged

    # Initial evaluations
    x1 = b - (b - a) * inv_phi
    x2 = a + (b - a) * inv_phi

    loss1, traj1, conv1 = evaluate_full(x1)
    loss2, traj2, conv2 = evaluate_full(x2)

    best_loss = min(loss1, loss2)
    best_N = int(round(x1)) if loss1 <= loss2 else int(round(x2))
    best_traj = traj1 if loss1 <= loss2 else traj2

    if conv1 or conv2:
        if verbose:
            print(f"      ✓ Candidate converged! Stopping search early.")
        return best_N, best_traj, best_loss

    for iteration in range(max_iter):
        if (b - a) <= 2.0:
            if verbose:
                print(f"      Interval converged to [{a:.1f}, {b:.1f}]")
            break

        if loss1 < loss2:
            b, x2, loss2, traj2, conv2 = x2, x1, loss1, traj1, conv1
            x1 = b - (b - a) * inv_phi
            loss1, traj1, conv1 = evaluate_full(x1)
        else:
            a, x1, loss1, traj1, conv1 = x1, x2, loss2, traj2, conv2
            x2 = a + (b - a) * inv_phi
            loss2, traj2, conv2 = evaluate_full(x2)

        if loss1 < best_loss:
            best_loss = loss1
            best_N = int(round(x1))
            best_traj = traj1
        if loss2 < best_loss:
            best_loss = loss2
            best_N = int(round(x2))
            best_traj = traj2

        if conv1 or conv2:
            if verbose:
                print(f"      ✓ Candidate converged at iteration {iteration+1}! Stopping search early.")
            return best_N, best_traj, best_loss

        if verbose:
            print(f"Iteration {iteration+1}: interval=[{a:.1f}, {b:.1f}], best N={best_N}, loss={best_loss:.6f}")

    if verbose:
        print(f"Best N={best_N} (loss={best_loss:.6f})")

    return best_N, best_traj, best_loss


def run_full_backward(device, forward_model, val_data, num_samples, verbose,
                      N_fixed=138, save_dir=Path('refinement_results')):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    results = []

    for idx in tqdm(range(min(num_samples, len(val_data)))):
        stencil, target, traj_gt, true_N, params = val_data[idx]
        stencil = stencil.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        print(f'\n{"="*60}')
        print(f'Sample {idx+1}/{num_samples} | Dataset N = {true_N} (using fixed N = {N_fixed})')
        print(f'{"="*60}')

        # 1. Generate ONE analytical prior (using fixed N)
        init_traj = generate_prior_trajectory(
            stencil, target, params, N_fixed,
            N_theta=15, N_phi=20, N_r=100, phi_max=0.1, height=1000,
            pixel_size=5e-3, L=500e-3, temperature=1.0
        ).to(device)

        # 2. Refine trajectory (full 2000 steps, no early stopping)
        refined_traj, final_loss, _ = refine_trajectory(
            forward_model, stencil, target, init_traj, N_fixed, device,
            target_loss=None,          # no early stop
            max_steps=2000,
            patience=30,
            verbose=verbose
        )

        # 3. Final physics simulation on refined trajectory
        final_sim_ms_ssim = None
        final_sim_np = None
        try:
            traj_np = refined_traj[0].cpu().numpy()
            traj_phys = []
            for theta_norm, phi_norm in traj_np:
                theta_phys = theta_norm * 2 * np.pi
                phi_phys = phi_norm * 0.1
                traj_phys.append([theta_phys, phi_phys])

            stencil_obj = reconstruct_stencil_from_params(params)
            final_sim_target = run_ground_truth_simulation(
                stencil_obj, traj_phys, diffusion=params['diffusion'],
                point_count=N_fixed
            )
            final_sim_np = final_sim_target
            target_np = target[0,0].cpu().numpy()

            target_tensor = torch.tensor(target_np).float().unsqueeze(0).unsqueeze(0).to(device)
            sim_tensor = torch.tensor(final_sim_np).float().unsqueeze(0).unsqueeze(0).to(device)
            final_sim_ms_ssim = ms_ssim(target_tensor, sim_tensor).item()

            print(f"  Final simulation MS-SSIM: {final_sim_ms_ssim:.4f}")
        except Exception as e:
            print(f"  Final simulation failed: {e}")

        # Store results
        results.append({
            'idx': idx,
            'true_N': true_N,
            'used_N': N_fixed,
            'final_loss': final_loss,
            'final_ms_ssim': final_sim_ms_ssim,
        })

        # Plotting: 2x2 (stencil, prior?, GT target, simulation result)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Top left: stencil
        axes[0,0].imshow(stencil[0,0].cpu().numpy(), cmap='viridis')
        axes[0,0].set_title('Stencil')
        axes[0,0].axis('off')
        
        # Top right: forward model prediction after refinement (optional)
        with torch.no_grad():
            final_pred = forward_model({
                'stencil': stencil,
                'traj': [refined_traj[0]],
                'num_pts': torch.tensor([N_fixed], device=device)
            })
        axes[0,1].imshow(final_pred[0,0].cpu().numpy(), cmap='viridis')
        axes[0,1].set_title(f'Refined Prediction (N={N_fixed})')
        axes[0,1].axis('off')
        
        # Bottom left: ground truth target
        axes[1,0].imshow(target[0,0].cpu().numpy(), cmap='viridis')
        axes[1,0].set_title('Ground Truth Target')
        axes[1,0].axis('off')
        
        # Bottom right: physics simulation of refined trajectory
        if final_sim_np is not None:
            axes[1,1].imshow(final_sim_np, cmap='viridis')
            axes[1,1].set_title(f'Simulation (MS-SSIM={final_sim_ms_ssim:.3f})')
        else:
            axes[1,1].text(0.5, 0.5, 'Simulation Failed', ha='center', va='center')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{idx}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Final summary
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    ms_ssim_vals = [r['final_ms_ssim'] for r in results if r['final_ms_ssim'] is not None]
    if ms_ssim_vals:
        print(f"Mean MS-SSIM: {np.mean(ms_ssim_vals):.4f} ± {np.std(ms_ssim_vals):.4f}")
    print(f"Average final loss: {np.mean([r['final_loss'] for r in results]):.6f}")
    
    # Save JSON summary
    import json
    with open(save_dir / 'results_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
    
print("Loading forward model...")
forward_model = UNet()
forward_model = forward_model.to(device)
load_weights(model=forward_model, path='/teamspace/studios/this_studio/custom_2_pt7.pth')
forward_model.eval()
    
print("Loading validation data...")
val_data = SimulationDataset_Backward(
    file_path='/teamspace/studios/this_studio/random_point_dataset_validation', 
    train=True
    )
print(f"Loaded {len(val_data)} validation samples")
     
    
    
run_full_backward(
    device=device, 
    forward_model=forward_model, 
    val_data=val_data, 
    num_samples=30, 
    verbose=True,
    save_dir=Path('refinement_results')
)

