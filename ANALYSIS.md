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

---

## 6. Feasibility & strategy: is the forward CV model the right tool?

This section addresses the strategic question directly: *does a learned
forward surrogate make sense for this problem at all, and where should the
learning effort actually go (property prediction, 3D curvature matching)?*

### 6.1 The uncomfortable observation: the current physics is linear

In `mbhl/simulation.py` the forward map is

```
F(trajectory) = histogram2d of displacements  R_i = (D+δ)·tan(φ_i)·(cos θ_i, sin θ_i)
                then Gaussian diffusion blur                    (generate_F, lines 384-471)
Deposition    = fftconvolve(stencil M, F)                       (simulate_fftconvolve)
```

So the deposition is **exactly a sum of shifted copies of one fixed kernel**
`K = M ⊛ G_diffusion`:

```
D(x) = Σ_i K(x − R_i)
```

It is *linear in the trajectory point measure* and *smooth in each point's
(θ, φ)*. Two consequences:

1. **A differentiable physics forward model is ~50 lines of PyTorch**, not a
   25M-parameter U-Net. Precompute `K̂(k) = M̂(k)·Ĝ(k)` once per stencil,
   then `D̂(k) = K̂(k)·Σ_i exp(−i k·R_i)` via the Fourier shift theorem —
   exact forward values, exact analytic gradients w.r.t. every (θ_i, φ_i),
   no training data, no generalization error, batch-parallel on GPU, and
   trivially correct against the numpy implementation.
2. **The U-Net is being asked to memorize a convolution.** All the
   architectural machinery (set attention, periodic coordinate maps,
   count gating) exists to approximate an operation the simulator performs
   exactly in milliseconds. The surrogate can only ever be *worse* than the
   physics here — the usual surrogate justifications (speed, or
   differentiability of an otherwise black-box simulator) do not hold,
   because the simulator is already fast *and* analytically differentiable.

**Recommendation:** for inverse design against the *current* physics,
replace the learned forward model with a differentiable-physics
reimplementation and keep the rest of the student's pipeline unchanged —
the analytical prior → gradient refinement → N-search structure carries
over verbatim, just with exact gradients instead of surrogate gradients.
Estimated effort: 1–2 weeks including numerical validation against `mbhl`.
This also eliminates the entire class of surrogate-hallucination failure
modes during optimization (the optimizer exploiting regions where the
network extrapolates badly — a known risk that the current pipeline
mitigates only by the final physics check).

### 6.2 Where a learned model *does* make sense

The surrogate becomes the right tool exactly where the cheap linear physics
stops being the truth. In this repo those places are already visible:

- **Shadowing and thickness effects** — the SI notebooks
  (`SI-fig19-shadowing`, `SI-fig21/22-shadowing-error`,
  `SI-fig23-thickness-increase`) quantify systematic *nonlinear* deviations
  from the ideal convolution model (aperture shadowing at large φ,
  progressive stencil clogging). A network that learns the *residual*
  between ideal convolution and shadowing-corrected simulation (or
  experiment) is a genuinely useful surrogate: small, data-efficient, and
  it composes with the exact linear term.
- **Sim-to-real (AFM morphology)** — `fig4-AFM-simulation` compares
  simulation to measured AFM data. A learned map from ideal deposition
  → measured 3D morphology is the highest-value learning problem in this
  project, because *no* cheap physics exists for it. This is also the
  natural home for the student's encoder work.
- **Amortized inverse design** — a network that maps target → trajectory
  directly (set transformer / conditional diffusion over point sets) makes
  sense once many inverse queries must be answered fast. That is a
  months-scale research project and only worth it after the
  optimization-based pipeline works end to end.

### 6.3 Property-prediction head on the learned space

Adding a feed-forward property head on the shared encoder (bottleneck →
MLP → scalar/vector properties) is cheap (days of work) and multi-task
training may regularize the representation. But apply a simple test first:

> **If the property is computable from the deposition/height field, don't
> learn it — compute it differentiably downstream.**

Mean/Gaussian curvature of the 2.5D height field, feature width, contrast,
connectivity proxies, spectral content — all are closed-form (Sobel-like
derivative stencils for curvature) and can be implemented as fixed
differentiable PyTorch ops on top of the (exact or learned) forward output.
A learned head only earns its place when the property is (a) expensive to
compute (full optical/mechanical response requiring FDTD/FEM), or (b) only
observable experimentally (measured AFM curvature, adhesion, optical
scattering). In case (b) the head should hang off *measured-data*
training, i.e. the sim-to-real model of §6.2 — that is where the learned
space genuinely pays off.

### 6.4 Inverse design against properties / 3D curvature

Currently the backward pipeline matches the target **image** (geometry).
Upgrading to property/curvature matching is structurally easy once the
forward map is differentiable end to end:

```
trajectory → forward (physics or surrogate) → height field
           → differentiable property extractor (curvature, etc.)
           → loss vs target property → ∇ back to trajectory
```

- With the differentiable physics of §6.1 plus a curvature operator
  (second-derivative filters on the height field), this is **feasible now**
  at roughly the same effort as fixing the current pipeline (~2–3 weeks
  total). One practical caveat: the current deposition map is a *dose*
  field resized to 256²; converting dose → physical height (including the
  thickness-increase correction from SI-fig23) needs to be part of the
  chain before curvature is physically meaningful.
- **Ill-posedness is the real new difficulty**, not differentiability: a
  low-dimensional property target (e.g. "mean curvature = X on the ridges")
  admits many trajectories. Expect the optimizer to find degenerate
  solutions unless regularized. The student's analytical prior is exactly
  the right countermeasure and generalizes: keep an image-space or
  prior-likelihood term in the loss as a regularizer alongside the property
  term, and/or optimize within the four parametric trajectory families of
  `Dataset.py` (few parameters, physically realizable, machine-executable)
  instead of free point clouds. Free-point optimization also produces
  trajectories a real stage may not be able to execute — a smoothness /
  max-slew penalty is cheap to add.
- **True 3D curvature** (undercuts, resist sidewalls, growth dynamics —
  beyond a 2.5D height field) is not reachable from the current simulation
  layer at all; that requires either extending the physics or the
  sim-to-real model of §6.2. Treat it as a separate project phase.

### 6.5 Suggested effort allocation

| Option | Effort | Risk | Verdict |
|---|---|---|---|
| Fix current surrogate pipeline to running state (§5 items 1–2) | ~1 week | low | Do regardless — it is the student's baseline and the comparison point for the paper. |
| Differentiable-physics forward + property extractors | 1–2 weeks | low | **Do first.** Likely obsoletes the surrogate for current-physics inverse design; exact gradients, zero data. |
| Property head on existing encoder (sim-derived properties) | days | low | Skip unless the property is expensive — compute it downstream instead (§6.3). |
| Property/curvature-matched inverse design (2.5D) | 2–3 weeks on top of the above | medium (ill-posedness) | High value; reuse the analytical prior as regularizer. |
| Learned residual for shadowing/thickening | weeks, needs data gen | medium | Good student project; first place the surrogate beats pure physics. |
| Sim-to-real AFM morphology model | months, needs measured data | high | Biggest scientific payoff; where the CV/encoder investment truly belongs. |
| Amortized inverse network (target → trajectory) | months | high | Defer until the optimization-based pipeline is a working baseline. |

**Bottom line:** the student's forward U-Net, as an approximation of the
current linear physics, is hard to justify on technical grounds — a
differentiable reimplementation of the physics dominates it for inverse
design. But the *pipeline around it* (analytical prior, refinement loop,
physics-in-the-loop validation) is the durable contribution and transfers
unchanged. The learned-model effort should be redirected to where physics
is genuinely missing — shadowing/thickening residuals and especially
AFM-measured 3D morphology — and property/curvature matching should be
built as differentiable operators on top of the forward map rather than as
new learning problems.

---

## 7. Proposed architecture for the differentiable forward model

Premise: the stencil representation is a free choice (full 2D image, or
parametric lattice + aperture). Under that freedom, the recommended design
is a **spectral (Fourier-multiplier) physics core with an optional neural
residual operator on top** — three layers, each independently useful.

### 7.1 Layer 0 — closed-form spectral physics core

The entire ideal forward map factorizes as a *product of closed-form
Fourier multipliers*, because the deposition is a convolution chain:

```
D̂(k) = M̂(k) · Ĝ(k) · T̂(k)                 then one inverse FFT
        │        │        │
        │        │        └─ trajectory factor:  T̂(k) = Σᵢ wᵢ·exp(−i k·Rᵢ)
        │        │           Rᵢ = (D+δ)·tan(φᵢ)·(cos θᵢ, sin θᵢ) + drift
        │        └─ diffusion factor:  Ĝ(k) = exp(−|k|²σ²/2)
        └─ stencil factor (either representation, see below)
```

Every factor is analytic and smooth in **all** physical parameters, so a
PyTorch implementation gives exact autograd gradients w.r.t. trajectory
points (θᵢ, φᵢ), per-point dwell weights wᵢ, diffusion σ, drift, *and* the
stencil parameters — enabling joint stencil+trajectory co-design, which the
current pipeline cannot do at all.

The stencil factor `M̂(k)` is where the representation freedom lives, and
the spectral core is agnostic to the choice:

- **Parametric (recommended default).** A periodic lattice of identical
  apertures factorizes further into *form factor × structure factor*:
  `M̂(k) = A(k; shape) · Σⱼ exp(−i k·cⱼ)` over the basis centers cⱼ of the
  unit cell. The form factors are textbook closed forms — circle:
  jinc `2πr²·J₁(|k|r)/(|k|r)`; rectangle/square: `w·h·sinc(kₓw/2)·sinc(k_y h/2)`
  with rotation applied by rotating k. All four lattices in `Dataset.py`
  (square / hexagonal / honeycomb / diamond) are just different basis-center
  tables — exactly the structure the student already encoded. Smooth in
  r, w, h, rotation, and lattice constant L. ~5 designable scalars, always
  manufacturable, no rasterization anywhere.
- **Image / free-form.** When topology freedom is wanted (topology-
  optimization-style stencil design), represent the aperture by a signed
  distance function or density field ρ and soft-binarize
  `M = σ((−SDF)/τ)` with temperature annealing on τ, then FFT. Standard
  practice from differentiable rendering / topology optimization; needs
  a manufacturability regularizer (minimum feature size = a cap on |∇ρ|
  or an opening/closing penalty).
- Since both routes just produce `M̂(k)`, they can coexist behind one
  interface; start parametric, add free-form only if a design study needs
  it.

Bonus corrections over the current numpy implementation, for free:

- `generate_F` histograms displacements into pixel bins
  (`simulation.py:449-467`) — the phase factor `exp(−i k·Rᵢ)` places points
  with *exact sub-pixel* positions, removing quantization noise from both
  values and gradients.
- The FFT path quantizes diffusion to whole pixels
  (`sigma = int(diffusion/h)`, `simulation.py:469`) — `Ĝ(k)` uses the exact
  continuous σ (and note the other code paths at lines 709/787 already use
  float σ, so the current package is internally inconsistent here).
- Brillouin-zone folding becomes exact circular convolution on the unit
  cell (periodic FFT), rather than explicit index folding.

Cost: one batched inverse FFT per forward; microseconds on GPU; ~50–100
lines. Validate against `mbhl` numerically (expect agreement to within the
histogram/σ quantization of the original).

### 7.2 Layer 1 — differentiable trajectory parameterization

For inverse design, optimize not only free point clouds but the compact
parametric families already defined in `Dataset.py` (`smooth`, `stepwise`,
`harmonic`): Fourier coefficients of φ(θ), segment levels, etc. All map
smoothly into the Rᵢ of Layer 0. Free points maximize expressiveness but
produce stage-unexecutable trajectories; the parametric families are the
regularization (§6.4) *and* the manufacturability constraint in one. Dwell
weights wᵢ (continuous, positive, simplex-normalized) subsume the discrete
"number of points N" search — replacing golden-section over integer N with
smooth optimization + sparsity penalty on w.

### 7.3 Layer 2 — neural residual operator (only where physics ends)

For shadowing/thickening/sim-to-real (§6.2), learn only the *residual*
around the exact linear term:

```
D = D_linear  +  f_θ( D_linear, conditioning maps )
```

Architecture choice for f_θ: a **Fourier Neural Operator (FNO)** — the
natural pick because the ideal operator *is* a Fourier multiplier, so FNO's
spectral convolutions contain the truth as a special case and the network
only has to learn the deviation; it is also resolution/discretization-
invariant, which fits the resize-to-256² pipeline and lets one model serve
multiple mesh densities. A plain shallow CNN/U-Net residual is an
acceptable simpler fallback. Conditioning channels should carry the physics
the residual actually depends on: per-pixel incident-angle maps (the
φ-weighted dose from Layer 0's factorization), aperture edge distance/
orientation (from the SDF — another argument for the parametric/SDF
stencil), and cumulative dose for thickening dynamics. If the stencil is
parametric, its ~5 scalars enter via FiLM-style modulation of the residual
blocks — a hypernetwork is overkill at this parameter count.

Layer 2 is trained on shadowing-corrected simulations or AFM data
(hundreds of samples suffice for a residual, vs. the tens of thousands a
full surrogate needs), and is simply omitted until those datasets exist —
Layers 0–1 alone already replace the current U-Net for inverse design.

### 7.4 Why not the alternatives

- **Full neural surrogate (current U-Net, or FNO trained end-to-end):**
  learns what Layer 0 computes exactly; pays data, training, and
  hallucination costs for negative accuracy benefit (§6.1).
- **DeepONet / hypernetwork over stencil params:** sensible for parametric-
  only stencils, but strictly less general than the multiplier
  factorization, which handles image stencils through the same interface.
- **Differentiable ray tracer** (mirroring `simulate_raytracing`): correct
  but slow and gradient-noisy; only needed if membrane-thickness effects at
  large φ must be exact, and even then better handled as a Layer 2 residual
  trained on ray-traced data.
- **Autodiff through the existing numpy code** (e.g. via JAX rewrite of
  `mbhl`): workable, but the histogram binning and `int(σ)` quantization
  are non-differentiable/piecewise-constant operations that would have to
  be replaced anyway — at which point one has rebuilt Layer 0.

---

## 8. Salvage value of the image2image model: the latent space

Replacing the U-Net as the *forward operator* (§6–7) does not write off the
training investment. A model trained on tens of thousands of
(stencil, trajectory) → deposition pairs has learned a representation of
MBHL pattern space, and that representation has several concrete second
lives. The honest framing: **the forward-surrogate training was simulation
pretraining** — the product is not the predicted image, it is the encoder.

One caveat frames everything below. The latent was trained on the *ideal
linear physics*, so any knowledge it encodes is a lossy compression of what
the spectral core computes exactly. Its value is therefore never "it knows
the physics" — it is (a) a data-efficient starting point for tasks where
labels are scarce, and (b) a structured similarity metric over pattern
space. Ranked by value-per-effort:

1. **Pretrained trunk for the data-scarce tasks (highest value).** The
   shadowing residual and AFM sim-to-real models of §6.2 will have
   hundreds, not tens of thousands, of samples. Fine-tuning from the
   forward-trained encoder (`enc1–enc3`, and the per-point trajectory
   embeddings from `traj_gte`/`combined_gte` for set-conditioned residuals)
   is exactly the transfer-learning pattern that makes such sample counts
   workable. The student's `load_weights` (shape-checked partial loading)
   and `freeze_for_settransformer` (encoder freezing) were visibly built
   for this workflow already. This is the strongest argument that the work
   was scaffolding, not waste.

2. **Domain-specific perceptual loss in the inverse loop (cheapest win).**
   Pixel MSE is a poor metric for periodic patterns (translation-sensitive,
   blind to structure); MS-SSIM only partly compensates. Reusing the frozen
   encoder as a feature-matching loss — compare encoder activations of
   prediction vs. target, VGG-perceptual-loss style but native to
   lithography patterns — drops into `refine_trajectory` with ~10 lines and
   works identically on top of the spectral physics core. This directly
   serves the property/curvature-matching goal (§6.4), where good pattern
   metrics matter more than pixel agreement.

3. **Retrieval-based initialization, complementing the analytical prior.**
   Embed the training set once (pooled bottleneck vectors), and at inverse-
   design time retrieve nearest neighbors of the target pattern — their
   stored trajectories are initializations that come from the *data*
   rather than from the prior's geometric assumptions (single central
   aperture, clean ring), so they cover exactly the cases where the
   analytical prior degrades (overlapping deposition from multiple
   apertures, dense honeycomb interference). A day of work with the
   existing checkpoint.

4. **Out-of-distribution / trust scoring during optimization.** Latent
   distance to the training manifold flags when an optimizer has pushed a
   design into territory where the *learned residual* (Layer 2) — and later
   the experimental calibration — cannot be trusted. Added as a soft
   penalty, it keeps property-matched inverse design (which is ill-posed
   and will exploit any model, §6.4) inside the validated envelope. Note
   this is a safeguard for the learned components, not for the exact
   physics core, which needs none.

5. **Latent generative prior over stencils (only if free-form design
   happens).** If stencil design ever goes beyond the ~5-parameter lattices
   to free-form topology (§7.1), a generative model (VAE/diffusion) in or
   near this latent space provides the manufacturability prior that
   free-form optimization otherwise lacks. Real value, but contingent on a
   design-space decision that has not been made — do not build it
   speculatively.

Two practical notes. First, the U-Net latent is spatially structured
(32×32×256 bottleneck), which is ideal for the dense tasks (1) and (2);
global uses (3)–(4) should pool it — the existing `stencil_token` head is
that pooling. Second, if representation quality ever becomes the goal in
itself, masked-autoencoder or contrastive pretraining on the simulation
data would likely beat forward-map pretraining — but the forward-trained
checkpoint is already paid for, and items (1)–(4) can be built on it as-is.

**Bottom line:** keep the checkpoint, retire the *role*. The network's job
was never to out-compute an FFT; its encoder is the down payment on the
experiment-facing models where learning is genuinely needed, and its latent
space supplies the pattern-similarity metric, initialization source, and
trust region that the exact-physics inverse pipeline lacks on its own.
