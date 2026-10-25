# PINN Groundwater Surrogate (Optuna + TDR-RAR)

A **Physics-Informed Neural Network (PINN)** surrogate model for transient
groundwater flow, using **Optuna** for hyperparameter tuning and **TDR-RAR**
(Time-Dependent Residual–based Residual Adaptive Refinement) for dynamic
collocation-point generation. The network is built with
[SciANN](https://www.sciann.com/) and is trained to emulate a 7-layer
MODFLOW model of the GM-47 block while honouring the governing flow physics and
field observations.

## Table of contents

- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Quickstart](#quickstart)
- [Computational requirements](#computational-requirements)
- [Outputs](#outputs)
- [Documentation](#documentation)
- [License](#license)

## Features

- 3-D (7-layer) transient PINN with hard constraints for specified-head
  boundaries and physics residuals for flow, storage, vertical leakance,
  pumping wells and stream leakage.
- **Optuna** hyperparameter optimization (network size, learning rate, loss
  weights, collocation sampling ratios) with median pruning.
- **TDR-RAR** dynamic refinement: collocation points are re-sampled over
  training cycles toward regions of high time-dependent PDE residual.
- Evaluation against CGWB and daily monitoring wells, with MODFLOW as the
  baseline, plus publication-ready figures.
- Fully **portable paths** — no machine-specific locations are hard-coded;
  configure everything via command-line flags or environment variables.

## Repository structure

```
TDR-RAR/
├── training/
│   └── optuna_tdr_rar_pinn_training.py   # end-to-end: build → (tune) → train → evaluate → report
├── evaluation/
│   ├── evaluate_final_model_cpu.py       # score a trained checkpoint (CPU)
│   └── evaluate_final_model_gpu.py       # score a trained checkpoint (GPU)
├── visualization/
│   ├── plot_evaluation_metrics.py
│   ├── plot_rar_spatial_distribution.py
│   ├── plot_water_budget_comparison.py
│   ├── plot_hydraulic_parameters.py
│   └── plot_boundary_errors.py
├── data/
│   ├── generate_synthetic_data.py        # writes a runnable synthetic sample dataset
│   └── README.md                         # full data schema
├── docs/
│   ├── USAGE.md                          # inputs, outputs, options for every script
│   └── TUTORIAL.md                       # step-by-step first run
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

- **training/** — the core `optuna_tdr_rar_pinn_training.py` script. Implements
  both Phase 1 (Optuna hyperparameter search) and Phase 2 (TDR-RAR dynamic
  refinement loops), then evaluates and reports.
- **evaluation/** — scripts that evaluate the final trained checkpoints across
  both PDE physical errors and spatial observation (MAE / RMSE) errors.
- **visualization/** — scripts that generate the deliverable plots, including
  the spatial RAR distributions, water-budget component bar charts and loss
  metrics.

## Requirements

- Python 3.9 (3.8–3.10 supported)
- TensorFlow **2.10.x** (last release with native-Windows GPU support)
- SciANN, NumPy (< 1.24), pandas, SciPy, scikit-learn, h5py
- [FloPy](https://github.com/modflowpy/flopy) and pyproj (MODFLOW I/O + CRS)
- Matplotlib, Pillow, python-docx, openpyxl, Optuna

Exact pinned versions are in [`requirements.txt`](requirements.txt) and
[`environment.yml`](environment.yml). The key constraint is TensorFlow 2.10 with
`numpy < 1.24` and a compatible SciANN build.

## Installation

Using conda (recommended; also provides the CUDA runtime for GPU use):

```bash
conda env create -f environment.yml
conda activate pinn_tdr_rar
```

Or using pip in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Windows + GPU:** the CUDA DLLs bundled in a conda environment live in
`<env>/Library/bin`. Point the training script at them with `--conda-env` or
the `PINN_CONDA_ENV` environment variable. This is a no-op on Linux/macOS or CPU.

## Data

The model inputs are a MODFLOW simulation of the GM-47 block plus two
observation-well datasets. **The field/model data are confidential and are
available from the authors on request.**

So the pipeline can be run and tested without that data, a generator produces a
**synthetic** sample dataset with the identical structure, shapes and units
(the numbers are plausible fabrications, not measurements):

```bash
python data/generate_synthetic_data.py --outdir ./sample_workspace
```

The expected files and their full schema are documented in
[`data/README.md`](data/README.md):

```
<workspace>/
├── GM47_b1_MODFLOW/{GM47_b1.h5, GM47_b1.hed, GM47_b1.ccf}
├── well_heads_CGWB.csv
└── well_heads_daily.csv
```

**Reviewers / third parties:** a fully randomized, non-confidential test model
(identical structure, no real measurements) is described in
[`test_model/README.md`](test_model/README.md) — generate it with one command,
or download the ready-made archive from the
[Releases](../../releases) page, then run the pipeline against it to confirm the
code executes without errors.

## Quickstart

```bash
# 1. set up the environment (see Installation)
conda activate pinn_tdr_rar

# 2. generate the synthetic sample data
python data/generate_synthetic_data.py --outdir ./sample_workspace

# 3. train (point the script at the workspace)
python training/optuna_tdr_rar_pinn_training.py --workspace ./sample_workspace

# 4. evaluate and visualize
export PINN_WORKSPACE=$PWD/sample_workspace
python evaluation/evaluate_final_model_cpu.py
python visualization/plot_rar_spatial_distribution.py
```

See [`docs/TUTORIAL.md`](docs/TUTORIAL.md) for a guided first run and
[`docs/USAGE.md`](docs/USAGE.md) for every script's inputs, outputs and options.

## Computational requirements

The original experiments were run on:

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA Quadro RTX 4000 (8 GB VRAM) |
| System memory | 128 GB RAM |
| CUDA / cuDNN | 11.2 / 8.1 (bundled via `environment.yml`) |
| OS | Windows (native-GPU TensorFlow 2.10) |
| Runtime | ~8–10 hours for a full run **including** Optuna hyperparameter tuning |

Notes:

- The model fits within 8 GB of VRAM. A CUDA GPU with ≥ 8 GB is recommended;
  training also runs on CPU, but substantially slower.
- System RAM matters because the full MODFLOW grid, stress-period arrays and
  collocation sets are held in memory; 16–32 GB is enough for the synthetic
  sample case, while the full dataset benefits from more.
- Without hyperparameter tuning (`run_optuna = False`, the default), a single
  final training run is considerably shorter than the full 8–10 h figure.

## Outputs

Training and evaluation write to the workspace: trained weights
(`pinn_surrogate_weights.h5`, `checkpoint_weights_cycle_*.h5`), loss history
(`loss_history.csv`/`.png`), Optuna history (`optuna_history.csv`), per-dataset
evaluation CSVs, scatter and spatial-error comparison figures, mid-period head
GeoTIFFs, and a `final_report.md` summary. The full list is in
[`docs/USAGE.md`](docs/USAGE.md).

## Documentation

- [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — end-to-end walkthrough with the
  synthetic data.
- [`docs/USAGE.md`](docs/USAGE.md) — inputs, outputs, options and configuration
  for every script.
- [`data/README.md`](data/README.md) — data schema and the synthetic-data
  generator.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
