from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import box
from tqdm.auto import tqdm
from shapely.affinity import rotate 

from mbhl import *
from mbhl.geometry import Geometry, Mesh, Circle, Rectangle, Square 
from mbhl.simulation import Stencil, Physics, System 
from mbhl.utils import sqrt2, sqrt3
from mbhl.utils import nm, um
from math import sqrt 
from skimage.transform import resize  
import scipy.ndimage as ndi 
import cv2
import random



def Universal_Honeycomb_Lattice(r=None, L=None, w=None, h=None, type='circle', orientation='vertical', rotation=0):

    orientation=orientation.lower()
    type=type.lower()
    
    assert orientation in ("vertical", "horizontal"
        ), "Orientation must be either vertical or horizontal"
    assert type in (
        "circle", "rectangle", "square"
        ), "Type must be either circle or rectangle or square"
    
    

    lattice_configs = {
        "vertical": {
            "cell": (L*sqrt3, L*3),
            "centers": [(0,0), (L*sqrt3/2, L/2), (L*sqrt3/2, L*3/2), (L*sqrt3, L*2)]
        }, 
        "horizontal": {
            "cell": (L*3, L*sqrt3), 
            "centers":[(0,0), (L/2, L/2*sqrt3), (L*3/2, L/2*sqrt3), (L*2,L*sqrt3)]
        }
    }
    config=lattice_configs[orientation]
    patches=[]
    for cx, cy in config["centers"]:
        if type=="circle":
            patch=Circle(cx, cy, r)
        elif type=="square": 
            patch=Square(cx-w/2, cy-w/2, w)
        else: 
            patch=Rectangle(cx-w/2, cy-h/2, w, h)
        if type != 'circle' and rotation != 0: 
            patch=rotate(patch, rotation, origin=(cx, cy))
        patches.append(patch)

    cell=config["cell"]
    return Geometry(patches=patches, cell=cell, pbc=(True, True))

def Universal_Hexagonal_Lattice(r=None, w=None, h=None, L=None, type='circle', orientation='vertical', rotation=0):
    orientation=orientation.lower()
    type=type.lower()

    assert orientation in ("vertical", "horizontal"
        ), "Orientation must be either vertical or horizontal"
    assert type in (
        "circle", "rectangle", "square"
        ), "Type must be either circle or rectangle or square"

    

    lattice_configs = {
        "vertical": {
            "cell": (L*sqrt3, L*3),
            "centers": [(0,0), (L*sqrt3/2, L/2), (L*sqrt3/2, L*3/2), (L*sqrt3/2, L*5/2), (L*sqrt3, L), (L*sqrt3, L*2)]
        }, 
        "horizontal": {
            "cell": (L*3, L*sqrt3), 
            "centers":[(0,0), (L/2, L/2*sqrt3), (L*3/2, L/2*sqrt3), (L*5/2,L*sqrt3/2), (L, L*sqrt3), (L*2, L*sqrt3)]
        }
    }

    config=lattice_configs[orientation]
    patches=[]
    for cx, cy in config["centers"]:
        if type=="circle":
            patch=(Circle(cx, cy, r))
        elif type=="square": 
            patch=(Square(cx-w/2, cy-w/2, w))
        else: 
            patch=(Rectangle(cx-w/2, cy-h/2, w, h))
        if type != 'circle' and rotation != 0: 
            patch=rotate(patch, rotation, origin=(cx, cy))
        patches.append(patch)

    cell=config["cell"]
    return Geometry(patches=patches, cell=cell, pbc=(True, True))

def Universal_Square_Lattice(r=None, w=None, h=None, L=None, type='circle', rotation=0): 
    type=type.lower()
    
    assert type in (
        "circle", "rectangle", "square"
        ), "Type must be either circle or rectangle or square"
    
    

    cell=(L, L)
    
    patches=[]

    if type=='circle':
        patch=(Circle(0, 0, r))
    elif type=='square': 
        patch=(Square(w/2 * -1, w/2 * -1 , w))
    else: 
        patch=(Rectangle(w/2 * -1, h/2 * -1 , w, h))
    if type != 'circle' and rotation != 0: 
        patch=rotate(patch, rotation, origin=(0, 0))
    patches.append(patch)

    return Geometry(patches=patches, cell=cell, pbc=(True, True))

def Universal_Diamond_Lattice(r=None, w=None, h=None, L=None, type="circle", rotation=0):
    type=type.lower()
    
    assert type in (
        "circle", "rectangle", "square"
        ), "Type must be either circle or rectangle or square"
    
    
    
    cell=(L*sqrt2, L*sqrt2)

    patches=[]

    coordinates=[(0,0), (L*sqrt2/2, L*sqrt2/2)]

    for cx, cy in coordinates: 
        if type=='circle': 
            patch=(Circle(cx, cy, r))
        elif type=='square': 
            patch=(Square(cx-w/2, cy-w/2, w))
        else: 
            patch=(Rectangle(cx-w/2, cy-w/2, w, h))
        if type != 'circle' and rotation != 0: 
            patch=rotate(patch, rotation, origin=(cx, cy))
        patches.append(patch)
    
    return Geometry(patches=patches, cell=cell, pbc=(True, True) )


def GenerateRLRatio(r=None, w=None, h=None, type='square', L=None): 
    type=type.lower() 
    if type=='square': 
        diagonal=(np.sqrt(2)*w)/2 
    if type=='circle': 
        diagonal=r
    if type=='rectangle': 
        diagonal=sqrt(w**2+h**2) / 2 
    max_RL=0.5-diagonal/L-10*nm/L

    ratio=np.random.uniform(0, max_RL)
    RL_ratio=np.maximum(0.01, ratio)
    return RL_ratio 
        
def CheckInputs(r=None, L=None, w=None, h=None, type=None): 
    type=type.lower()
    assert type in (
        "circle", "rectangle", "square"
        ), "Type must be either circle or rectangle or square"
    
    if type=='circle':
        if r>L/2: #should always pass
            return False 
    elif type=='square': 
        diagonal=sqrt(2*(w**2))
        if diagonal > L: 
            return False 
    elif type=='rectangle': 
        diagonal=sqrt(w**2+h**2)
        if diagonal > L: 
            return False 
    return True 



################Per Point Dataset########################
def RandomTraj(phi_max, n_pts):
    theta = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    theta = theta + np.random.uniform(0, 2*np.pi/n_pts)  # random phase offset
    
    traj_class = np.random.choice(
        ['independent', 'smooth', 'stepwise', 'harmonic'],
        p=[0.4, 0.2, 0.2, 0.2]
    )
    
    if traj_class == 'independent':
        # Current training distribution
        theta = np.random.uniform(0, 2*np.pi, n_pts)
        phi   = np.random.uniform(0.001, phi_max, n_pts)
    
    elif traj_class == 'smooth':
        # Ellipse-like — smooth phi(theta)
        # amplitude sampled to stay within phi_max
        n_harmonics = np.random.randint(1, 3)
        amplitude   = np.random.uniform(0.1, 0.5)  # fraction of mean
        phases      = np.random.uniform(0, 2*np.pi, n_harmonics)
        mean_phi    = np.random.uniform(0.001, phi_max * 0.7)
        
        phi = np.ones(n_pts) * mean_phi
        for i in range(n_harmonics):
            phi += amplitude * mean_phi * np.cos((i+1) * theta + phases[i])
        phi = np.clip(phi, 0.001, phi_max)
    
    elif traj_class == 'stepwise':
        # Dwell-like — step function phi(theta)
        n_segments = np.random.randint(2, 6)
        boundaries = np.sort(np.random.uniform(0, 2*np.pi, n_segments-1))
        boundaries = np.concatenate([[0], boundaries, [2*np.pi]])
        levels     = np.random.uniform(0.001, phi_max, n_segments)
        
        phi = np.zeros(n_pts)
        for i in range(n_segments):
            mask      = (theta >= boundaries[i]) & (theta < boundaries[i+1])
            phi[mask] = levels[i]
    
    elif traj_class == 'harmonic':
        # High_amp-like — high frequency variation
        n_waves   = np.random.randint(3, 6)
        amplitude = np.random.uniform(0.3, 0.6)
        phases    = np.random.uniform(0, 2*np.pi, n_waves)
        mean_phi  = np.random.uniform(0.001, phi_max * 0.5)
        
        unit = np.ones(n_pts)
        for i in range(n_waves):
            unit += amplitude/n_waves * np.cos((i+1) * theta + phases[i])
        unit = np.maximum(unit, 0.1)
        phi  = unit * mean_phi
        phi  = np.clip(phi, 0.001, phi_max)
    
    return np.column_stack([theta, phi])



def PerPointDataset_Random(phi_max_limit, n_samples, run_name, drive_folder='forward_model_point_dataset'):
    base_path = Path("/teamspace/studios/this_studio")
    save_folder = base_path / drive_folder 
    save_folder.mkdir(parents=True, exist_ok=True)
    
    nm = 1e-9
    um = 1e-6
    L, H = 500*nm, 5*um
    default_h = 5*nm
    h_ratio = 1.0

    for i in range(1, n_samples+1):
        n_pts = random.randint(a = 1, b = 40)
        valid_geometry = False 
        while not valid_geometry: 
            lattice = np.random.choice(['hexagonal', 'square', 'honneycomb','diamond'])
            shape_type = np.random.choice(['square', 'circle', 'rectangle'])
            orientation = np.random.choice(['vertical', 'horizontal'])
            
            w, r, height = None, None, None 
            if shape_type == 'square': 
                w = np.random.uniform(100, 200)*nm
                Rl_ratio = GenerateRLRatio(w=w, L=L, type='square')
            elif shape_type == 'circle': 
                r = np.random.uniform(100, 200)*nm
                Rl_ratio = GenerateRLRatio(r=r, L=L, type='circle')
            else: 
                w = np.random.uniform(100, 200)*nm
                height = np.random.uniform(75, 150)*nm
                Rl_ratio = GenerateRLRatio(w=w, h=height, L=L, type='rectangle')
            
            valid_geometry = CheckInputs(r=r, w=w, L=L, h=height, type=shape_type)

        diffusion = np.random.uniform(10, 20)*nm
        rotation = np.random.uniform(45, 90)

        #generate trajectory: 
        trajectory = RandomTraj(phi_max = phi_max_limit, n_pts = n_pts)


        # ALWAYS fold to first Brillouin zone – kernel size = unit cell, extra=1
        fold_to_bz = True
        phys = Physics(trajectory, diffusion=diffusion, drift=0)

        # Lattice Selection Logic
        lattice_func = {
            'hexagonal': Universal_Hexagonal_Lattice,
            'square': Universal_Square_Lattice,
            'diamond': Universal_Diamond_Lattice,
            'honneycomb': Universal_Honeycomb_Lattice
        }[lattice]
        
        lat_args = {'r':r, 'L':L, 'w':w, 'h':height, 'type':shape_type, 'rotation':rotation}
        if lattice in ['hexagonal', 'honneycomb']: 
            lat_args['orientation'] = orientation
        
        stencil = Stencil(lattice_func(**lat_args), thickness=0, gap=H, h=default_h/h_ratio)
        system = System(stencil=stencil, physics=phys)
        
        # Prepare matrices – with fold_to_bz=True, extra=1, M_padded is 3×3 tiled unit cell
        F, M_padded, M_origin, recover_indices, extra = system._prepare_matrices(
            add_diffusion=True, fold_to_bz=fold_to_bz
        )
        system.simulate(method="fft", fold_to_bz=fold_to_bz)
        
        # target_raw is already 3×3 tiled because extra=1
        target_raw = system.results.tiled_mesh(extra_x=extra).array

        # Resize both to 256×256 – M_padded and target_raw are already 3×3 tiled
        input_stencil = resize(M_padded, (256, 256), order=0)
        target_ready = resize(target_raw, (256, 256), order=1, anti_aliasing=True)

        # Save everything – recover_indices are no longer needed but kept for compatibility
        np.savez_compressed(
            save_folder / f"sim_{i:04d}_{run_name}.npz",
            input_stencil=input_stencil,
            target_deposition=target_ready,
            unit_cell_coords=np.array([0, 256, 0, 256]),  # placeholder
            lattice=lattice,
            shape=shape_type,
            RL_ratio=Rl_ratio, 
            trajectory=trajectory, 
            rotation=rotation, 
            width=w, 
            radius=r, 
            height=height, 
            orientation=orientation, 
            diffusion=diffusion, 
        )
        if i % 100 == 0: 
            print(f"Saved {i}/{n_samples}")


