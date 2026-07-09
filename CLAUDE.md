# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this repository is

Computational package for **Molecular-Beam Holographic Lithography (MBHL)**,
associated with a published manuscript (Zenodo DOI 10.5281/zenodo.14986964).
It contains two largely independent layers:

1. **`mbhl/` — the installable physics package** (stable, published):
   geometry primitives built on `shapely` (`mbhl/geometry.py`), the
   stencil/physics/system simulation engine (`mbhl/simulation.py`, FFT-based
   convolution of a stencil pattern with a molecular-beam trajectory filter),
   Fourier helpers, and 3D model building for Blender (`mbhl/build.py`).
   `setup.py` installs only this package. The notebooks under
   `manuscript_figs/` reproduce the paper figures using it.

2. **Root-level ML research scripts** (exploratory, student work, *not*
   packaged): `NanoLithoForwardModel.py`, `AnalyticalPipeline.py`,
   `Dataset.py`. These implement a CV/deep-learning surrogate of the forward
   simulation and a backward (inverse-design) pipeline on top of it. They were
   copied out of a Lightning AI studio (`/teamspace/studios/this_studio/...`
   paths) and do **not** run as-is in this repo — see "Known issues" below.

## The ML layer at a glance

- **`Dataset.py`** — dataset *generation*. Builds randomized stencil
  geometries (square / hexagonal / honeycomb / diamond lattices with circle /
  square / rectangle apertures), samples trajectories from four families
  (`independent`, `smooth`, `stepwise`, `harmonic`), runs the `mbhl` FFT
  simulation, and saves `(input_stencil, target_deposition, trajectory,
  params)` as 256×256 `.npz` samples.

- **`NanoLithoForwardModel.py`** — the **forward surrogate model**: an
  attention-augmented U-Net (`UNet`) that maps
  `(stencil image [1×256×256], trajectory point set [N×2])` →
  predicted deposition pattern `[1×256×256]`. Key components:
  `GlobalTrajEncoder` (weight-tied residual MLP for per-point embeddings),
  `PhysicsSetAttention` (softmax-free cross-attention from spatial pixels to
  the trajectory point set), `MultiScaleStencil` + `CustomAttention`
  (multi-scale stencil tokens injected in the decoder), `PeriodicCoordGen`
  (learned periodic coordinate maps), and `AttentionGate` skip connections
  gated by point count. Also contains LR-finder/OneCycle utilities, several
  loss functions (`CustomLoss` = 0.8·MSE + 0.2·MS-SSIM is the one used),
  training/eval loops, and a module-level training/eval script.

- **`AnalyticalPipeline.py`** — the **backward (inverse) design pipeline**:
  given a target deposition pattern, (1) find the central aperture in the
  stencil, (2) extract an angular intensity histogram p(θ) and conditional
  radial→φ CDFs p(φ|θ) around it, (3) sample an initial trajectory from this
  analytical prior, (4) refine it by gradient descent through the frozen
  forward `UNet` (Adam on the trajectory itself, MSE+MS-SSIM loss), with an
  optional golden-section search over the number of points N, and (5)
  validate the refined trajectory with the real `mbhl` physics simulation.

Normalization conventions used throughout the ML code: trajectory columns are
`(θ, φ)`; θ is normalized by `2π`, φ by `0.1` rad (hard-coded `phi_max`);
deposition targets are divided by the trajectory point count.

## Known issues (do not "fix" silently — see ANALYSIS.md first)

- `AnalyticalPipeline.py` imports `DatasetGeneration`, but the file in this
  repo is named `Dataset.py` → `ImportError` as checked in.
- `AnalyticalPipeline.py` calls the forward model with
  `{'stencil', 'traj': [tensor], 'num_pts'}` while `UNet.forward` expects
  `traj` to be a batched tensor — the two files reflect *different versions*
  of the model interface and are not currently compatible.
- Both `NanoLithoForwardModel.py` and `AnalyticalPipeline.py` execute
  training/evaluation at module scope (no `if __name__ == "__main__":`), with
  hard-coded `/teamspace/studios/this_studio/...` data and checkpoint paths.
  Importing `NanoLithoForwardModel` therefore attempts to load checkpoints
  and run a full validation pass.
- `torch`, `torchmetrics`, `scikit-image`, `opencv`, `tqdm` are required by
  the ML scripts but are not declared in `setup.py` or the README.

A detailed architecture description and line-referenced code review lives in
**`ANALYSIS.md`** at the repo root.

## Conventions & practical notes

- Python ≥ 3.8, `numpy < 2.0` (see README conda command). Install the physics
  package with `pip install -e .`.
- Pre-commit is configured (`.pre-commit-config.yaml`) — run
  `pre-commit run --files <changed files>` before committing.
- The `mbhl` package and manuscript notebooks back a published paper: treat
  them as frozen unless explicitly asked; changes there can break figure
  reproduction.
- The root ML scripts are research artifacts. Prefer refactoring them into a
  proper subpackage (e.g. `mbhl/ml/`) with CLI entry points over patching
  module-level script code in place.
- There are no automated tests in the repo; validation is done through the
  manuscript notebooks and the evaluation routines inside the ML scripts.
