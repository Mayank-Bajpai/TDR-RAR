# Randomized test model (for reproducibility / code checks)

This is a **fully randomized**, non-confidential copy of the model inputs,
provided so a third party can verify that the training, evaluation and
visualization code runs end-to-end **without the confidential field data**.

Every numeric value — hydraulic properties, simulated heads, fluxes (recharge,
ET, stream leakage, well pumping), boundary heads, and the observation-well
levels and coordinates — is machine-generated and randomized. It contains
**none** of the real GM-47 measurements or monitoring-well locations. Only the
*structure* the code requires is preserved (grid 105 × 217, 7 layers, 21 stress
periods, 110 time steps, and the file/array layout documented in
[`../data/README.md`](../data/README.md)).

> It is a test fixture for "does the code run?" checks only. The numbers are not
> physically meaningful and must not be interpreted scientifically.

## How to get it

This folder already contains the parts small enough to track in git:
`GM47_b1_MODFLOW/GM47_b1.h5` and the two `well_heads_*.csv` files. The two large
binaries (`GM47_b1.hed` ≈ 134 MB, `GM47_b1.ccf` ≈ 39 MB) exceed GitHub's
file-size limits and are **not** tracked here — get them either way below.

### Option A — generate it (recommended, reproducible)

The generator ships in the repo and builds the complete dataset with one
command. Use **`--seed 2025`** so the output matches the files already committed
in this folder (the generator will (re)write all five files consistently).

```bash
pip install -r requirements.txt          # or: pip install numpy pandas h5py flopy pyproj
python data/generate_synthetic_data.py --outdir ./test_model --seed 2025
```

Then run the pipeline against it:

```bash
export PINN_WORKSPACE=$PWD/test_model     # Windows: setx PINN_WORKSPACE "%CD%\test_model"
python training/optuna_tdr_rar_pinn_training.py --workspace ./test_model
```

### Option B — download the ready-made archive

A pre-built archive, `TDR-RAR_test_model.zip` (~82 MB), is attached to the
repository's **[Releases](../../releases)** page. Download, unzip, and point the
pipeline at the unzipped folder as above.

> **Why a Release and not a tracked file?** The binary head file
> (`GM47_b1.hed`) is ~134 MB, which exceeds GitHub's 100 MB per-file limit for
> files committed into the repository. Release assets have no such limit, so the
> ready-made copy lives there; Option A reproduces it locally either way.

## What's in the dataset

```
test_model/
├── GM47_b1_MODFLOW/
│   ├── GM47_b1.h5     (~7 MB)    grid + hydraulic properties + boundary packages   [tracked here]
│   ├── GM47_b1.hed    (~134 MB)  binary heads   (flopy HeadFile, double precision)  [generate / Release]
│   └── GM47_b1.ccf    (~39 MB)   binary budget  (flopy CellBudgetFile, double)      [generate / Release]
├── well_heads_CGWB.csv                                                              [tracked here]
└── well_heads_daily.csv                                                             [tracked here]
```

`[tracked here]` files are committed in this folder. `[generate / Release]`
files are too large for GitHub to store in the repository (its browser uploader
rejects files over 25 MB and the repository blocks files over 100 MB), so they
come from the generator (Option A, `--seed 2025`) or the Release (Option B).

See [`../data/README.md`](../data/README.md) for the full field-by-field schema.

## Verification

The generated dataset has been checked to load through the exact
`flopy` / `h5py` / observation-ingestion code paths the scripts use: the head
file reads back as `(110, 7, 105, 217)`, the budget file returns the
`RECHARGE`, `ET` and `STREAM LEAKAGE` records the code expects, the H5 groups
and shapes match, and all observation rows survive the coordinate-transform and
active-grid filter. Running the full training loop additionally requires the
pinned TensorFlow 2.10 environment from [`../requirements.txt`](../requirements.txt).
