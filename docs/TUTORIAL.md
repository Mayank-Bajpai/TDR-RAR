# Tutorial: run the pipeline end-to-end

This walkthrough takes you from a fresh clone to trained-model outputs using the
built-in **synthetic** sample data, so you can verify the whole pipeline without
the confidential field dataset. Everything here runs the same way on the real
data — you just point `PINN_WORKSPACE` at the real workspace instead.

Estimated time: a few minutes to set up; the full training run is long (see
step 5). For a quick check, a reduced run finishes in minutes on CPU.

## 1. Clone and enter the repository

```bash
git clone https://github.com/Mayank-Bajpai/TDR-RAR.git
cd TDR-RAR
```

## 2. Create the environment

With conda (recommended — handles the CUDA runtime for GPU):

```bash
conda env create -f environment.yml
conda activate pinn_tdr_rar
```

Or with pip in a virtual environment (CPU, or an existing CUDA setup):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Windows + GPU:** the CUDA DLLs shipped in a conda environment live in
> `<env>/Library/bin`. Tell the training script where to find them with
> `--conda-env` or `PINN_CONDA_ENV`, e.g.
> `--conda-env "C:\Users\you\anaconda3\envs\pinn_tdr_rar"`.
> On Linux/macOS or CPU you can ignore this.

## 3. Generate the synthetic sample dataset

```bash
python data/generate_synthetic_data.py --outdir ./sample_workspace
export PINN_WORKSPACE=$PWD/sample_workspace        # Windows: setx PINN_WORKSPACE "%CD%\sample_workspace"
```

This writes `GM47_b1.h5`, `GM47_b1.hed`, `GM47_b1.ccf` and the two
`well_heads_*.csv` files into `sample_workspace/`. See
[`../data/README.md`](../data/README.md) for what each file contains.

## 4. (Optional) quick smoke test

Confirm the data loads and the model builds before committing to a long run.
A short run is easiest via the in-script toggle `final_epochs` near the top of
`training/optuna_tdr_rar_pinn_training.py` — set it to a small number such as
`20`, keep `run_optuna = False`, then:

```bash
python training/optuna_tdr_rar_pinn_training.py --workspace ./sample_workspace
```

You should see it load the grid, load the MODFLOW heads/budget, ingest the
observations, build the SciANN model, and start printing training loss.

## 5. Full training run

Restore `final_epochs` (e.g. `500`) and, if you want hyperparameter tuning, set
`run_optuna = True`:

```bash
python training/optuna_tdr_rar_pinn_training.py --workspace ./sample_workspace
```

On the reference hardware (NVIDIA Quadro RTX 4000, 8 GB VRAM; 128 GB system
RAM) a complete run including Optuna tuning takes roughly **8–10 hours**. The
model fits comfortably in 8 GB of VRAM but can also run on CPU (much slower).

Outputs land in the workspace: trained weights, `loss_history.csv/png`,
evaluation CSVs, comparison figures, and `final_report.md`.

## 6. Evaluate and visualize

```bash
python evaluation/evaluate_final_model_cpu.py        # scores the saved checkpoint
python visualization/plot_rar_spatial_distribution.py
```

(The evaluation scripts expect a `checkpoint_weights_cycle_9_saved.h5`; training
produces cycle checkpoints you can rename/point to — see
[`USAGE.md`](USAGE.md).)

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not load dynamic library 'cudart64_110.dll'` (Windows) | Pass `--conda-env`/`PINN_CONDA_ENV` so the CUDA DLLs are on the path, or run on CPU. |
| `FileNotFoundError: ...GM47_b1.h5` | Set `PINN_WORKSPACE` (or `--workspace`) to the folder that contains `GM47_b1_MODFLOW/`. |
| TensorFlow can't see the GPU | Check the TF 2.10 ↔ CUDA 11.2 ↔ cuDNN 8.1 versions; `environment.yml` pins a known-good set. |
| `numpy` ABI / dtype errors | Ensure `numpy < 1.24` (required by TensorFlow 2.10). |
| Very few observations after filtering | Expected with the synthetic data; the real datasets have many more wells. |
