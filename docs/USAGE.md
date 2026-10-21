# User guide: inputs, outputs and options

This guide documents how to configure and run each script, what it reads, and
what it writes. For a step-by-step first run, see
[`TUTORIAL.md`](TUTORIAL.md).

## Configuring paths (no hard-coded locations)

All machine-specific paths are supplied at run time. Nothing is hard-coded to a
particular user or drive.

| Setting | CLI flag (training) | Environment variable | Default |
|---------|--------------------|----------------------|---------|
| Workspace (data in, results out) | `--workspace` | `PINN_WORKSPACE` | current directory (training) / repo root (evaluation & plots) |
| MODFLOW model folder | `--model-dir` | `PINN_MODEL_DIR` | `<workspace>/GM47_b1_MODFLOW` |
| Conda env with CUDA DLLs (Windows GPU only) | `--conda-env` | `PINN_CONDA_ENV` | unset (no-op) |

The evaluation and visualization scripts read the `PINN_WORKSPACE` environment
variable (falling back to the repository root), so the same workspace can be
shared across all stages:

```bash
export PINN_WORKSPACE=/path/to/your/workspace      # Linux/macOS
# setx PINN_WORKSPACE "C:\path\to\workspace"       # Windows (new shells)
```

## Input files

The workspace must contain the data described in [`../data/README.md`](../data/README.md):

```
<workspace>/
├── GM47_b1_MODFLOW/{GM47_b1.h5, GM47_b1.hed, GM47_b1.ccf}
├── well_heads_CGWB.csv
└── well_heads_daily.csv
```

Don't have the field data? Generate a synthetic sample case:

```bash
python data/generate_synthetic_data.py --outdir ./sample_workspace
export PINN_WORKSPACE=$PWD/sample_workspace
```

## 1. Training — `training/optuna_tdr_rar_pinn_training.py`

Runs the full pipeline: builds the SciANN PINN, (optionally) tunes
hyperparameters with Optuna, trains the final model with TDR-RAR dynamic
collocation refinement, evaluates it and writes a report.

```bash
python training/optuna_tdr_rar_pinn_training.py --workspace ./sample_workspace
python training/optuna_tdr_rar_pinn_training.py --help    # list all options
```

In-script toggles near the top of the configuration section:

| Variable | Effect |
|----------|--------|
| `run_optuna` | `True` runs the Optuna hyperparameter search; `False` uses the stored best parameters |
| `retrain_final` | `True` retrains the final model; `False` loads saved weights |
| `final_epochs` | Epochs for the final training run |

**Outputs** (written to the workspace): `pinn_surrogate_weights.h5`,
`checkpoint_weights_cycle_*.h5` / `_inputs_*.npz` / `_loss_*.csv`,
`loss_history.csv` + `loss_history.png`, `optuna_history.csv` (if tuning),
`*_evaluation_results.csv`, comparison figures (`*_comparison.png`,
`mid_timestep_comparison.*`, `spatial_error_comparison_daily.png`), GeoTIFFs of
mid-period heads, and `final_report.md`.

## 2. Evaluation — `evaluation/`

Loads a trained checkpoint and scores it against the observations and the
MODFLOW baseline.

- `evaluate_final_model_cpu.py` — forces CPU; reads `checkpoint_weights_cycle_9_saved.h5`.
- `evaluate_final_model_gpu.py` — uses the GPU if available; same checkpoint.

```bash
python evaluation/evaluate_final_model_cpu.py        # uses PINN_WORKSPACE / repo root
```

**Outputs:** per-dataset evaluation CSVs, scatter comparison plots (PNG/SVG/TIF),
`cycle_9_eval_metrics.json`, and mid-time-step head comparison figures.

## 3. Visualization — `visualization/`

Standalone plotting/reporting scripts. Each honours `PINN_WORKSPACE`.

| Script | Produces |
|--------|----------|
| `plot_evaluation_metrics.py` | Evaluation-metric summary figures |
| `plot_rar_spatial_distribution.py` | Spatial distribution of RAR collocation points |
| `plot_water_budget_comparison.py` | MODFLOW vs PINN water-budget bar charts (reads an Excel budget workbook passed to the function) |
| `plot_hydraulic_parameters.py` | Calibrated hydraulic-parameter maps |
| `plot_boundary_errors.py` | Boundary-error analysis report (.docx) |

Some visualization scripts expect auxiliary analysis files (e.g. a `drop_1000/`
subfolder or an Excel budget workbook) produced by earlier analysis runs; they
are optional extras, not part of the core train → evaluate path.

## Reproducibility notes

- Random seeds are fixed (`numpy` seed 42, RFF seed 42) for repeatable sampling.
- The default hyperparameters are the best trial from the original Optuna study
  (stored in the script's `else` branch).
- Exact numbers depend on the input data and on TensorFlow/CUDA versions; see
  the computational-requirements section in the top-level `README.md`.
