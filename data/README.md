# Data: schema and synthetic sample case

The experiments in this repository use a MODFLOW simulation of the **GM-47
block** together with two observation-well datasets. The field/model data are
confidential and **available from the authors on request**. To let anyone run,
test and understand the pipeline without that data, this folder provides a
generator that writes a **synthetic** dataset with exactly the same structure,
shapes, dtypes and units.

> The synthetic numbers are smooth, physically-plausible fabrications. They are
> for exercising the code only — not for scientific interpretation.

## Generate the sample case

```bash
python data/generate_synthetic_data.py --outdir ./sample_workspace
```

Then point the pipeline at that folder (see `../docs/USAGE.md`):

```bash
# Linux/macOS
export PINN_WORKSPACE=$PWD/sample_workspace
# or pass it explicitly
python training/optuna_tdr_rar_pinn_training.py --workspace ./sample_workspace
```

For a quick format smoke-test (small files) use `--n-times 4`. The default
(`--n-times 110`) reproduces the full-size case and writes ~0.4 GB of binaries,
which is why the data itself is **not** committed to the repository — only this
generator is.

## Expected files and schema

The scripts read the following, relative to the workspace directory:

```
<workspace>/
├── GM47_b1_MODFLOW/
│   ├── GM47_b1.h5     # grid, hydraulic properties, boundary packages
│   ├── GM47_b1.hed    # binary heads   (flopy HeadFile, precision='double')
│   └── GM47_b1.ccf    # binary budget  (flopy CellBudgetFile, precision='double')
├── well_heads_CGWB.csv
└── well_heads_daily.csv
```

Fixed model geometry (hard-coded in the scripts): **105 rows × 217 columns,
7 layers, 21 stress periods, 110 time steps**, cell size Δr ≈ 747.876 m,
Δc ≈ 747.983 m.

### `GM47_b1.h5`

Node ids are **1-based**: `node = k*(105*217) + r*217 + c + 1`.

| Path | Shape (flat) | Meaning / units |
|------|--------------|-----------------|
| `Arrays/top1` | 105·217 | Top-of-model elevation (m) |
| `Arrays/HK{1..7}` | 105·217 | Horizontal hydraulic conductivity per layer (m/d, > 0) |
| `Arrays/SS{1..7}` | 105·217 | Specific storage per layer (1/m) |
| `Arrays/SY{1..7}` | 105·217 | Specific yield per layer (–) |
| `Arrays/bot{1..7}` | 105·217 | Layer-bottom elevation (m) |
| `Arrays/ibound{1..7}` | 105·217 | Active-cell flag (> 0 active) |
| `Arrays/VANI{1..7}` | 105·217 | Vertical anisotropy ratio (–) |
| `Specified Head/02. Cell IDs` | n_chd | 1-based node ids of CHD cells |
| `Specified Head/07. Property` | (≥1, n_chd, 21) | CHD head per cell per stress period (m); read as `[0, idx, sp]` |
| `Well/02. Cell IDs` | n_wel | 1-based node ids of well cells |
| `Well/07. Property` | (≥1, n_wel, 21) | Pumping rate per well per stress period (m³/d, < 0 = extraction); read as `[0, idx, sp]` |
| `Stream/02. Cell IDs` | n_str | 1-based node ids of stream cells |
| `Stream/07. Property` | (≥3, n_str, ≥1) | `[1, idx, 0]` = streambed conductance (m²/d); `[2, idx, 0]` = streambed bottom elevation (m) |

### `GM47_b1.hed` (binary heads)

Standard MODFLOW double-precision head file. One record per (time step, layer),
each with header `kstp, kper, pertim, totim, text='HEAD', ncol, nrow, ilay`
followed by `nrow·ncol` float64 values. Read with
`flopy.utils.HeadFile(path, precision='double').get_alldata()` →
array `(110, 7, 105, 217)`. Inactive cells carry the sentinel `-999.99`.

### `GM47_b1.ccf` (binary cell budget)

MODFLOW compact cell-by-cell budget, double precision, read with
`flopy.utils.CellBudgetFile(path, precision='double')`. Per time step:

- `RECHARGE` — full array (IMETH 1), returned as `(1, 105, 217)` m³/d
- `ET` — full array (IMETH 1), returned as `(1, 105, 217)` m³/d
- `STREAM LEAKAGE` — list (IMETH 2), a record array with fields `node`
  (1-based) and `q` (m³/d)

### `well_heads_CGWB.csv` and `well_heads_daily.csv`

Plain CSV with one row per observation. Required columns (others are ignored):

| Column | Meaning |
|--------|---------|
| `Long` | Longitude, decimal degrees (EPSG:4326) |
| `Lat` | Latitude, decimal degrees (EPSG:4326) |
| `DateTime` | Observation date (parseable by `pandas.to_datetime`) |
| `Ground Water Level (m)` | Observed water level (m); kept only if 0 ≤ value ≤ 120 |

Coordinates are projected to EPSG:32644 (UTM 44N) and mapped to grid
(row, col) via the affine transform embedded in the scripts, then filtered to
the active model domain and to days 0–638 after the reference date
(2021-10-31).
