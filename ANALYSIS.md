# Analysis of the CV-Based Forward / Backward Design Models

Reviewer: Claude (AI code review), July 2026.
Scope: the isolated root-level scripts `NanoLithoForwardModel.py` and
`AnalyticalPipeline.py` (with `Dataset.py` as supporting context), i.e. the
student's deep-learning work layered on top of the published `mbhl` physics
package.

---

## 1. What the system does

The goal is a differentiable surrogate for the MBHL simulation so that
lithography trajectories can be *designed* (inverse problem) rather than only
*simulated* (forward problem).

```
                    ┌──────────────────────────────────────────────┐
 Dataset.py         │ mbhl physics (FFT simulation)                │
 (data generation)  │ stencil geometry + trajectory → deposition   │──► .npz samples
                    └──────────────────────────────────────────────┘
                                        │ train
                                        ▼
 NanoLithoForwardModel.py   UNet(stencil image, trajectory point set)
 (forward surrogate)                → predicted deposition (256×256)
                                        │ frozen, differentiable
                                        ▼
 AnalyticalPipeline.py      analytical prior p(θ), p(φ|θ) from target
 (backward design)          → sample init trajectory
                            → gradient refinement THROUGH the UNet
                            → (optional) golden-section search over N
                            → validate with real mbhl simulation
```

This is a sound and fairly modern research design: physics-informed
data generation, a conditional image-to-image surrogate, and
optimization-through-the-surrogate for inverse design, with the real
simulator kept in the loop as the final referee. The *concept* is good; the
*engineering* has significant problems (Section 4).

---

## 2. Forward model architecture (`NanoLithoForwardModel.py`)

### 2.1 Overall topology

`UNet` (line 349) is a 3-level encoder/decoder U-Net over the 256×256
stencil image, heavily augmented with conditioning on the trajectory
point set. Channel flow:

- **Encoder** (`enc1/enc2/enc3`, lines 368–380): 1→32→64→256, stride-2
  convs, SiLU activations. Bottleneck at 32×32×256.
- **Decoder** (`dec1..dec3`, lines 397–427): bilinear upsample + conv
  stacks back to 256×256, output head `out_conv` (64→1) initialized with
  tiny gain 0.01 (line 430) — a sensible choice for a near-zero-mean
  residual-style target.
- **Skip connections** are *gated*, not plain concatenations
  (`AttentionGate`, line 193): encoder features are modulated by a
  sigmoid mask computed from encoder+decoder features, then multiplied by a
  global `gate` scalar derived from the point count
  (`sigmoid(log(num_pts))`, lines 472–474). The intent — trust the stencil
  skip features more when many trajectory points average out noise — is a
  nice physics-motivated inductive bias.

### 2.2 Trajectory conditioning (the interesting part)

- `GlobalTrajEncoder` (line 135): projects each (θ, φ) point to an
  embedding, then applies a **weight-tied** residual MLP block `num_layers`
  times (the same `mlp_block` is reused each iteration — this is recurrent
  weight sharing, not a stack of distinct layers; if unintentional it
  silently reduces capacity).
- Each point embedding is fused with a global stencil token
  (`stencil_token`, pooled from the bottleneck) and re-encoded to a 512-dim
  per-point embedding (`combined_gte`, lines 459–467), so every trajectory
  point "knows" what stencil it acts on.
- `PhysicsSetAttention` (line 221) then lets every spatial pixel attend
  over the set of point embeddings at three decoder scales. **Notably there
  is no softmax** — raw scaled dot-product scores are divided by N and used
  directly (lines 244–246). This makes the aggregation *additive* in the
  points (closer to how deposition physically accumulates per point) rather
  than a convex combination. If deliberate, this deserves a comment and a
  name in the paper; if accidental (a forgotten `F.softmax`), it still
  trains but is a different operator than "attention" implies.
- `MultiScaleStencil` (line 256) produces stencil feature maps at 4 scales
  which are injected into the decoder via `CustomAttention` cross-attention
  (`mst_attn1/2`), and `PeriodicCoordGen` (line 284) predicts a soft
  mixture over candidate lattice periods {16,32,64,128} and tiles
  normalized coordinate grids accordingly — a periodic CoordConv that
  matches the Brillouin-zone-folded, tiled nature of the targets. This is a
  thoughtful, domain-specific piece of design.

### 2.3 Losses and training

- `CustomLoss` = 0.8·MSE + 0.2·MS-SSIM (line 551) is what `train()` uses.
- Training loop (line 595): Adam + OneCycleLR, grad-norm clipping at 1.0,
  per-epoch image monitoring (`evaluate_monitor`), checkpointing gated on
  train loss < 0.1. Fine-tuning support: partial checkpoint loading with
  shape checking (`load_weights`, line 738 — genuinely useful) and encoder
  freezing (`freeze_for_settransformer`, line 759 — defined but never
  called).
- Hand-rolled `CLR` LR-finder and `OneCycle` classes (lines 15–100) are
  classic fastai-blog ports; `OneCycleLR` from PyTorch is also used, so the
  hand-rolled ones are mostly redundant baggage.

### 2.4 Data pipeline

`SimulationDataset` (line 317) loads `.npz` files, normalizes θ by 2π and φ
by a hard-coded 0.1, and divides the target by the point count (making the
target a *per-point* deposition, consistent with the additive attention
reading above). Fixed-seed shuffling gives a reproducible 80/20 split.
Caveat: the split is by file *after* a global shuffle — fine — but train and
val come from the same directory passed at two call sites, so changing the
directory contents silently changes the split.

---

## 3. Backward pipeline architecture (`AnalyticalPipeline.py`)

The inverse-design strategy is a three-stage pipeline:

1. **Analytical prior** (lines 130–362). From the target deposition
   pattern: find the centermost aperture (connected-component analysis,
   `find_aperture_centers`), collect intensity inside a search radius set by
   the lattice minimum distance (`ThetaSearch` → weighted angular histogram
   p(θ)), then per-θ-bin build a radial distribution mapped through
   `φ = arctan(r/height)` into conditional CDFs p(φ|θ)
   (`build_phi_distribution_per_theta`), and finally inverse-CDF sample N
   points with a temperature knob (`GeneratePrior`, `sample_from_cdf`).
   This is a smart use of the problem's geometry — the deposition ring
   around an aperture literally *is* an image of the (θ, φ) trajectory
   density — and it gives the optimizer a physically meaningful starting
   point instead of a random one.

2. **Gradient refinement** (`refine_trajectory`, line 365). The trajectory
   tensor itself is the optimization variable (`requires_grad_(True)`,
   Adam), pushed through the frozen forward UNet, with
   `BackwardLoss` = 0.9·MSE + 0.1·(1−MS-SSIM); clamped to [0,1], best-so-far
   tracking, patience-based early stop, optional target-loss stop.
   Optionally wrapped in a golden-section search over the point count N
   (`golden_search_with_full_refinement`, line 529) with memoized
   evaluations — although the actual entry point `run_full_backward`
   (line 618) bypasses the search and uses a fixed N=138.

3. **Physics validation** (lines 465–526, 652–677). The refined trajectory
   is de-normalized, the stencil is rebuilt from saved parameters
   (`reconstruct_stencil_from_params`), and the *real* `mbhl` simulation is
   run; MS-SSIM against the target is the reported end-to-end metric, with
   2×2 comparison figures and a JSON summary per run. Closing the loop with
   the true simulator instead of trusting the surrogate is exactly right
   methodologically.

---

## 4. Code quality assessment

**Overall: promising research prototype, not reproducible software.** The
modeling ideas are genuinely good and several utilities are well built, but
the scripts are checked in mid-experiment: they contain environment-specific
paths, dead code from abandoned iterations, and — most importantly — the two
files are **not mutually consistent** and cannot run in this repository.

### 4.1 Blocking defects (the code cannot run as checked in)

| # | Location | Issue |
|---|----------|-------|
| B1 | `AnalyticalPipeline.py:16` | `from DatasetGeneration import ...` — no `DatasetGeneration.py` exists in the repo; the generators live in `Dataset.py`. Immediate `ImportError`. |
| B2 | `AnalyticalPipeline.py:390-394, 698-702` | Calls `forward_model({'stencil':…, 'traj': [traj_c[0]], 'num_pts': …})` — `traj` as a Python **list** plus a `num_pts` key. `UNet.forward` (`NanoLithoForwardModel.py:442-445`) does `x_dict['traj'].shape` and never reads `num_pts`. The two files target **different versions of the model interface**; a list has no `.shape`, so refinement crashes at the first forward pass. |
| B3 | `NanoLithoForwardModel.py:727-798` | Training/eval code runs at **module scope** (no `if __name__ == "__main__":`). `AnalyticalPipeline.py:17` importing `UNet, load_weights` triggers checkpoint loading and a full `EvaluateValidation` run on import — against paths that only exist on the student's Lightning studio. |
| B4 | Both files | Hard-coded `/teamspace/studios/this_studio/...` paths for data, checkpoints, and outputs (`NanoLithoForwardModel.py:47,629-630,729,736`; `AnalyticalPipeline.py:747,752`). Nobody else can run this without editing source. |
| B5 | `NanoLithoForwardModel.py:538` | `MSSSIMLoss.__init__` reads the **global** `device` defined at line 728 — the class only works because of the module-level script below it; imported standalone it raises `NameError`. |

### 4.2 Correctness concerns (runs, but likely not what was intended)

- **`HybridLoss.forward` (`NanoLithoForwardModel.py:580-591`)**:
  `target_binary` is thresholded using the **prediction's** mask
  (`target_binary[pred > threshold_val] = 1`), so the Dice term compares the
  prediction's binarization with itself → Dice loss is identically 0 and
  contributes nothing. Also `pred_binary` is initialized to ones and then
  fully overwritten (dead init), and hard thresholding is non-differentiable
  so no gradient flows through the Dice term anyway. (Unused by `train()`,
  but a trap if anyone switches to it.)
- **`PhysicsSetAttention` has no softmax** (`NanoLithoForwardModel.py:244-246`)
  — see §2.2. Defensible as additive set aggregation, but undocumented; one
  comment line would prevent a future "fix" that changes model behavior.
- **`GlobalTrajEncoder` weight tying** (`NanoLithoForwardModel.py:151-157`)
  — `num_layers` iterations reuse one block's weights; and the call sites
  pass different `num_layers` (2 and 3) than the default 5, so effective
  depth is set far from the definition. Intentional? Comment it.
- **`AdaptiveMSELoss` (`NanoLithoForwardModel.py:543-549`)** just scales MSE
  by `n_pts²`; it does not change gradient *direction*, only the effective
  learning rate — probably not the intended "adaptive" behavior.
- **`EvaluateValidation` normalization (`NanoLithoForwardModel.py:676-678`)**
  normalizes `pred` with the **target's** min/max before MS-SSIM. That lets
  a prediction with the wrong absolute scale still score well; report at
  least one unnormalized metric alongside.
- **`train()` sets `model.diagnostic` (`NanoLithoForwardModel.py:606,617`)**
  but `UNet` never reads it — leftover from a removed debug path.
- **`SimulationDataset_Backward.__getitem__` (`AnalyticalPipeline.py:83-84`)**
  normalizes `traj` in place on the loaded array, then returns it — and
  `FilteredDataset.__init__` (line 44) *also* iterates the dataset,
  triggering the same loads. Harmless today only because each `np.load`
  returns fresh arrays; fragile if caching is ever added. Also note the
  returned `params['diffusion']` defaults (`15e-9`) silently mask missing
  keys.
- **Unit-system drift**: `Dataset.py` works in meters (`L=500*nm`), the
  backward prior in what appear to be micrometers (`L=500e-3`,
  `pixel_size=5e-3`, `height=1000` at `AnalyticalPipeline.py:426,639`) with
  no unit annotations. It works because only ratios matter in `ThetaSearch`,
  but `height=1000` (µm? px?) directly sets the φ mapping — one wrong
  assumption here skews every prior sample. Units belong in the parameter
  names or docstrings.
- **`golden_search_with_full_refinement` is dead in practice** — the driver
  uses fixed `N_fixed=138` (`AnalyticalPipeline.py:619,633`). Fine for an
  ablation, but the search — one of the pipeline's selling points — is
  currently unexercised, and golden-section assumes a unimodal loss in N,
  which is worth validating empirically.

### 4.3 Maintainability / hygiene

- **No `main()` guards, no CLI, no config**: every hyperparameter (batch
  size, lr schedule, N_fixed=138, phi_max=0.1, normalization constants) is a
  buried literal. The magic constant `0.1` for φ normalization appears
  independently in at least five places across three files — one changed
  copy breaks the pipeline silently.
- **Duplication**: `SimulationDataset` vs `SimulationDataset_Backward` are
  near-copies; the four `Universal_*_Lattice` generators share ~80% of their
  bodies (a single parameterized builder with a lattice→centers table would
  do); `cross_attention` (`NanoLithoForwardModel.py:125`) duplicates what
  `CustomAttention` does and is never called.
- **Dead code / unused imports**: `find_lr`/`CLR`/`OneCycle` instantiated
  but unused in the final flow (`NanoLithoForwardModel.py:795-796`);
  `freeze_for_settransformer` never called; `warp_polar`,
  `find_peaks(_cwt)`, `os`, `center_of_mass`, `find_objects` unused in
  `AnalyticalPipeline.py`; `cv2`, `ndi`, `box`, `Axes3D` unused in
  `Dataset.py`; `Mesh`/`Geometry` imports unused in places. `mask` is dead
  in `PeriodicCoordGen`? (no) — but `recover_indices`/`F`/`M_origin` from
  `_prepare_matrices` are unused in `run_ground_truth_simulation`.
- **Naming**: `Dataset.py` as a root-level module name collides conceptually
  with `torch.utils.data.Dataset`; `UNet` is far more than a U-Net (worth a
  descriptive name for the paper); mixed naming conventions
  (`EvaluateValidation` vs `evaluate_monitor`, `ThetaSearch` vs
  `sample_from_cdf`); typo `honneycomb` is load-bearing (it's the dict key
  *and* the saved dataset label — fixing the spelling later will orphan old
  data, so document it).
- **No tests, no requirements**: the ML stack (`torch`, `torchmetrics`,
  `scikit-image`, `opencv-python`, `tqdm`) is absent from `setup.py` and the
  README's conda line. A pinned `requirements-ml.txt` is the minimum; a
  smoke test that a random batch flows through `UNet` would have caught B2
  immediately.
- **Style**: pre-commit config exists but these files were clearly not run
  through it (trailing whitespace, inconsistent spacing, 3–4 blank-line
  runs, commented-out remnants like "IMPORTANT FIX").

### 4.4 What is genuinely good

- The **analytical prior** is the standout idea: it turns domain knowledge
  (deposition geometry ≈ trajectory density image) into an initialization
  that makes optimization-through-a-surrogate tractable. Publishable insight.
- **Physics-in-the-loop validation** — final MS-SSIM is computed against the
  *real* simulator, not the surrogate. Many papers skip this.
- Domain-motivated architecture pieces: `PeriodicCoordGen` (lattice-period
  coordinate maps), per-point target normalization + count-based skip
  gating, stencil-conditioned point embeddings.
- Practical utilities: shape-checked partial checkpoint loading
  (`load_weights`), memoized golden-section search, per-epoch visual
  monitoring, tiny-gain output initialization.

---

## 5. Recommended next steps (ordered)

1. **Make it importable**: rename/point the `DatasetGeneration` import at
   `Dataset.py` (or split `Dataset.py` into `datasets.py` + `lattices.py`),
   and wrap the module-level scripts in both files with
   `if __name__ == "__main__":` + argparse for paths.
2. **Reconcile the model interface** (B2): decide on one `forward()`
   signature — batched `traj` tensor (current `UNet`) — and update
   `refine_trajectory`/`run_full_backward` accordingly; add a 5-line smoke
   test that runs a random batch through the model and one refinement step.
3. **Centralize constants**: a small `config.py`/dataclass holding
   `PHI_MAX=0.1`, image size, normalization rules, unit conventions; import
   it everywhere the literals currently live.
4. **Delete dead code** (§4.3) and either fix or remove `HybridLoss`.
5. **Package it**: move the ML code to `mbhl/ml/` with `data/`, `models/`,
   `inverse/` modules; declare the extra dependencies as an
   `extras_require["ml"]` in `setup.py`.
6. **Document the deliberate oddities** (softmax-free attention, weight-tied
   encoder, per-point normalization) in docstrings — they read as bugs until
   explained, and they are the parts a reviewer will ask about.
