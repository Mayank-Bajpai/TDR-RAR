#!/usr/bin/env python3
"""
Generate a small **synthetic** sample dataset for the TDR-RAR PINN groundwater
surrogate so the pipeline can be run end-to-end without the confidential field
data (which is available from the authors on request).

The real model inputs are a MODFLOW-USG style simulation of the GM-47 block:

    GM47_b1_MODFLOW/GM47_b1.h5    grid + hydraulic properties + boundary packages
    GM47_b1_MODFLOW/GM47_b1.hed   binary heads  (flopy HeadFile, double precision)
    GM47_b1_MODFLOW/GM47_b1.ccf   binary budget (flopy CellBudgetFile, double)
    well_heads_CGWB.csv           observation wells (CGWB network)
    well_heads_daily.csv          observation wells (daily monitoring)

This script writes synthetic files with the SAME structure, shapes, dtypes and
units, filled with smooth, physically-plausible made-up numbers. They are NOT
real measurements and must only be used to exercise and test the code.

Usage
-----
    python data/generate_synthetic_data.py --outdir ./sample_workspace
    # then point the pipeline at it:
    #   PINN_WORKSPACE=./sample_workspace  (or --workspace ./sample_workspace)

The grid dimensions (105 x 217, 7 layers, 21 stress periods, 110 time steps)
match the hard-coded shapes in the training/evaluation scripts and should be
left at their defaults unless you also change those scripts. Use --n-times with
a small number only for a quick smoke test of the file format.

See data/README.md for the full schema description.
"""
import argparse
import os
import struct

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("h5py is required: pip install h5py") from exc

# --- Fixed model geometry (must match the pipeline scripts) ---
NROW, NCOL, NLAY = 105, 217, 7
NSP = 21                      # stress periods
DELR, DELC = 747.87628161225, 747.98335290138
PERLEN = [30.0, 31.0, 31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0,
          31.0, 30.0, 31.0, 31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 29.0]
# Coordinate reference used by the observation ingestion code
X0, Y0, DET = 573624.966, 2880613.56, 559399.08
AFFINE = np.array([[-225.254385, -713.147527],
                   [713.249611, -225.286629]])   # [row; col] = AFFINE @ [dx; dy] / DET
REF_DATE = pd.Timestamp("2021-10-31")


def node_id(k, r, c):
    """1-based global node id used throughout the H5 boundary packages."""
    return k * (NROW * NCOL) + r * NCOL + c + 1


def build_grid_arrays(rng):
    """Smooth, plausible per-layer hydraulic property fields."""
    yy, xx = np.mgrid[0:NROW, 0:NCOL].astype(np.float64)
    # Active-cell mask: an ellipse inside the domain (ibound > 0 where active)
    cy, cx = NROW / 2.0, NCOL / 2.0
    ell = ((yy - cy) / (NROW * 0.46)) ** 2 + ((xx - cx) / (NCOL * 0.46)) ** 2
    ibound_top = (ell <= 1.0).astype(np.int32)

    # Land-surface elevation sloping gently across the domain
    top1 = (120.0 + 0.03 * xx - 0.02 * yy).astype(np.float64)

    arrays = {"top1": top1}
    bot_prev = top1.copy()
    thickness = 25.0
    for l in range(1, NLAY + 1):
        # Hydraulic conductivity (m/d): log-normal-ish, strictly positive
        hk = (5.0 + 3.0 * np.sin(xx / 40.0) + 2.0 * np.cos(yy / 30.0)
              + 0.5 * rng.standard_normal((NROW, NCOL)))
        hk = np.clip(hk, 0.2, None)
        ss = np.full((NROW, NCOL), 1.0e-4 + 2.0e-5 * l)          # specific storage (1/m)
        sy = np.full((NROW, NCOL), 0.12 + 0.005 * l)             # specific yield (-)
        vani = np.full((NROW, NCOL), 10.0)                       # vertical anisotropy (-)
        bot = bot_prev - thickness
        ib = ibound_top.copy()                                   # same active footprint per layer
        arrays[f"HK{l}"] = hk.astype(np.float64)
        arrays[f"SS{l}"] = ss.astype(np.float64)
        arrays[f"SY{l}"] = sy.astype(np.float64)
        arrays[f"VANI{l}"] = vani.astype(np.float64)
        arrays[f"bot{l}"] = bot.astype(np.float64)
        arrays[f"ibound{l}"] = ib.astype(np.int32)
        bot_prev = bot
    return arrays, ibound_top


def boundary_packages(ibound_top, rng):
    """Build Specified Head (CHD), Well and Stream package arrays for the H5 file."""
    active_r, active_c = np.where(ibound_top > 0)

    # --- Specified Head on the outer ring of active cells, layer 0 ---
    ring = []
    for r, c in zip(active_r, active_c):
        if any(ibound_top[r + dr, c + dc] <= 0
               for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
               if 0 <= r + dr < NROW and 0 <= c + dc < NCOL):
            ring.append((r, c))
    ring = ring[::3]  # thin it out
    chd_ids = np.array([node_id(0, r, c) for r, c in ring], dtype=np.int32)
    # Property shape: (>=1, n_chd, NSP); code reads chd_prop[0, idx, sp]
    chd_prop = np.zeros((1, len(ring), NSP), dtype=np.float64)
    for idx, (r, c) in enumerate(ring):
        base = 95.0 + 0.01 * c - 0.008 * r
        chd_prop[0, idx, :] = base + 0.5 * np.sin(np.arange(NSP) / 3.0)

    # --- Wells: a scatter of pumping cells across layers ---
    n_wells = 40
    sel = rng.choice(len(active_r), size=n_wells, replace=False)
    wel_ids, wel_rows = [], []
    for s in sel:
        k = int(rng.integers(0, NLAY))
        wel_ids.append(node_id(k, active_r[s], active_c[s]))
        wel_rows.append(s)
    wel_ids = np.array(wel_ids, dtype=np.int32)
    wel_prop = np.zeros((1, n_wells, NSP), dtype=np.float64)
    # Negative rates = extraction (m^3/d), seasonally varying
    wel_prop[0, :, :] = -(200.0 + 150.0 * rng.random((n_wells, 1))) * \
        (1.0 + 0.3 * np.sin(np.arange(NSP) / 2.0))

    # --- Stream cells: a column of active cells near the centre, layer 0 ---
    mid_c = NCOL // 2
    str_cells = [(r, mid_c) for r in range(NROW) if ibound_top[r, mid_c] > 0]
    str_ids = np.array([node_id(0, r, c) for r, c in str_cells], dtype=np.int32)
    # Property shape: (>=3, n_str, >=1); code reads str_prop[1,idx,0] (conductance)
    # and str_prop[2,idx,0] (streambed bottom elevation)
    str_prop = np.zeros((3, len(str_cells), 1), dtype=np.float64)
    for idx, (r, c) in enumerate(str_cells):
        str_prop[1, idx, 0] = 500.0          # streambed conductance (m^2/d)
        str_prop[2, idx, 0] = 90.0 - 0.01 * r  # streambed bottom elevation (m)
    return (chd_ids, chd_prop, wel_ids, wel_prop, str_ids, str_prop,
            str_cells)


def write_h5(path, arrays, packages):
    (chd_ids, chd_prop, wel_ids, wel_prop, str_ids, str_prop, _str_cells) = packages
    with h5py.File(path, "w") as f:
        g = f.create_group("Arrays")
        for name, arr in arrays.items():
            # Stored flat (as in the original H5); the scripts reshape to (105, 217)
            g.create_dataset(name, data=arr.reshape(-1))
        sh = f.create_group("Specified Head")
        sh.create_dataset("02. Cell IDs", data=chd_ids)
        sh.create_dataset("07. Property", data=chd_prop)
        wl = f.create_group("Well")
        wl.create_dataset("02. Cell IDs", data=wel_ids)
        wl.create_dataset("07. Property", data=wel_prop)
        st = f.create_group("Stream")
        st.create_dataset("02. Cell IDs", data=str_ids)
        st.create_dataset("07. Property", data=str_prop)


def _pad16(text):
    b = text.encode("ascii")
    return b.ljust(16)[:16]


def write_head_file(path, heads, times):
    """
    Write a MODFLOW binary head file readable by
    flopy.utils.HeadFile(path, precision='double').

    heads : array (n_times, NLAY, NROW, NCOL), float64
    One record per (time, layer): header + NROW*NCOL doubles.
    """
    text = _pad16("HEAD")
    with open(path, "wb") as fh:
        for t, totim in enumerate(times):
            kper = int(np.searchsorted(np.cumsum(PERLEN), totim, side="right"))
            kper = min(kper, NSP - 1)
            for l in range(NLAY):
                hdr = struct.pack("<iidd16siii",
                                  1,              # kstp
                                  kper + 1,       # kper (1-based)
                                  float(totim),   # pertim
                                  float(totim),   # totim
                                  text,           # text (16 bytes)
                                  NCOL, NROW,     # ncol, nrow
                                  l + 1)          # ilay (1-based)
                fh.write(hdr)
                heads[t, l].astype("<f8").tofile(fh)


def _write_budget_record(fh, text, imeth, totim, nlay, payload):
    """Write one compact-budget record (nlay stored negative => compact)."""
    fh.write(struct.pack("<ii16siii",
                         1, 1,                 # kstp, kper (1-based, single SP label ok)
                         _pad16(text),
                         NCOL, NROW, -nlay))   # negative nlay flags compact budget
    fh.write(struct.pack("<iddd", imeth, 1.0, float(totim), float(totim)))
    if imeth == 1:
        payload.astype("<f8").tofile(fh)       # full (nlay, nrow, ncol) array
    elif imeth == 2:
        nodes, q = payload
        fh.write(struct.pack("<i", len(nodes)))
        rec = np.empty(len(nodes), dtype=[("node", "<i4"), ("q", "<f8")])
        rec["node"] = nodes
        rec["q"] = q
        rec.tofile(fh)
    else:
        raise ValueError(f"Unsupported imeth {imeth}")


def write_budget_file(path, times, recharge, et, stream_nodes, stream_q):
    """
    Write a MODFLOW compact cell-by-cell budget file readable by
    flopy.utils.CellBudgetFile(path, precision='double').

    recharge, et : arrays (n_times, NROW, NCOL)      -> written as imeth 1, nlay 1
    stream_nodes : 1-based node ids (layer 0)        -> imeth 2 list, per time
    stream_q     : array (n_times, n_stream)         -> stream leakage (m^3/d)
    """
    with open(path, "wb") as fh:
        for t, totim in enumerate(times):
            _write_budget_record(fh, "RECHARGE", 1, totim, 1,
                                  recharge[t][None, :, :])
            _write_budget_record(fh, "ET", 1, totim, 1,
                                  et[t][None, :, :])
            _write_budget_record(fh, "STREAM LEAKAGE", 2, totim, 1,
                                  (stream_nodes, stream_q[t]))


def make_observations(ibound_top, arrays, rng, n_cgwb=25, n_daily=60):
    """Create synthetic observation CSVs with the columns the pipeline expects."""
    try:
        import pyproj
        to_ll = pyproj.Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    except Exception:                                   # pragma: no cover
        to_ll = None

    active_r, active_c = np.where(ibound_top > 0)
    inv = np.linalg.inv(AFFINE)

    def rows_for(n):
        recs = []
        sel = rng.choice(len(active_r), size=n, replace=True)
        for s in sel:
            r, c = int(active_r[s]), int(active_c[s])
            dx, dy = (inv @ np.array([r, c])) * DET      # back out projected metres
            xs, ys = X0 + dx, Y0 + dy
            if to_ll is not None:
                lon, lat = to_ll.transform(xs, ys)
            else:
                lon, lat = 77.0 + xs * 1e-6, 20.0 + ys * 1e-6
            day = float(rng.integers(0, 639))
            dt = REF_DATE + pd.Timedelta(days=day)
            gwl = float(np.clip(30.0 + 10.0 * rng.standard_normal(), 0.5, 119.5))
            recs.append({"Station": f"SYN-{r:03d}-{c:03d}",
                         "Lat": round(lat, 6), "Long": round(lon, 6),
                         "DateTime": dt.strftime("%Y-%m-%d"),
                         "Ground Water Level (m)": round(gwl, 3)})
        return pd.DataFrame(recs)

    return rows_for(n_cgwb), rows_for(n_daily)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic TDR-RAR sample data")
    ap.add_argument("--outdir", default="./sample_workspace",
                    help="Target workspace directory (default: ./sample_workspace)")
    ap.add_argument("--n-times", type=int, default=110,
                    help="Number of head/budget time steps (default 110 = full case; "
                         "use a small value only for a quick format smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = os.path.abspath(args.outdir)
    model_dir = os.path.join(outdir, "GM47_b1_MODFLOW")
    os.makedirs(model_dir, exist_ok=True)
    n_times = args.n_times
    total_days = float(np.sum(PERLEN))
    times = np.linspace(total_days / n_times, total_days, n_times)

    print(f"Writing synthetic dataset to: {outdir}")
    print(f"  grid = {NROW} x {NCOL}, layers = {NLAY}, time steps = {n_times}")

    arrays, ibound_top = build_grid_arrays(rng)
    packages = boundary_packages(ibound_top, rng)
    (_chd_ids, _chd_prop, _wel_ids, _wel_prop,
     _str_ids, _str_prop, str_cells) = packages

    write_h5(os.path.join(model_dir, "GM47_b1.h5"), arrays, packages)
    print("  wrote GM47_b1.h5")

    # Synthetic heads: a smooth field (decreasing with depth + seasonal trend)
    # plus a randomized perturbation, so the values are made-up/non-confidential
    # while staying in a physically sensible range.
    yy, xx = np.mgrid[0:NROW, 0:NCOL].astype(np.float64)
    base = 95.0 + 0.01 * xx - 0.008 * yy
    heads = np.zeros((n_times, NLAY, NROW, NCOL), dtype=np.float64)
    for t in range(n_times):
        season = 1.0 * np.sin(2 * np.pi * times[t] / 365.0)
        for l in range(NLAY):
            heads[t, l] = (base - 1.5 * l + season
                           + 0.75 * rng.standard_normal((NROW, NCOL)))
    # Inactive cells carry MODFLOW's dry/no-flow sentinel
    inactive = ibound_top <= 0
    heads[:, :, inactive] = -999.99
    write_head_file(os.path.join(model_dir, "GM47_b1.hed"), heads, times)
    print("  wrote GM47_b1.hed")

    # Budget terms
    recharge = np.zeros((n_times, NROW, NCOL), dtype=np.float64)
    et = np.zeros((n_times, NROW, NCOL), dtype=np.float64)
    active = ibound_top > 0
    for t in range(n_times):
        recharge[t][active] = 2.0 * (1.0 + 0.4 * np.sin(2 * np.pi * times[t] / 365.0))
        et[t][active] = -0.8 * (1.0 + 0.3 * np.cos(2 * np.pi * times[t] / 365.0))
    stream_nodes = np.array([node_id(0, r, c) for r, c in str_cells], dtype=np.int32)
    stream_q = np.zeros((n_times, len(str_cells)), dtype=np.float64)
    for t in range(n_times):
        stream_q[t] = -50.0 * (1.0 + 0.2 * rng.standard_normal(len(str_cells)))
    write_budget_file(os.path.join(model_dir, "GM47_b1.ccf"),
                      times, recharge, et, stream_nodes, stream_q)
    print("  wrote GM47_b1.ccf")

    cgwb, daily = make_observations(ibound_top, arrays, rng)
    cgwb.to_csv(os.path.join(outdir, "well_heads_CGWB.csv"), index=False)
    daily.to_csv(os.path.join(outdir, "well_heads_daily.csv"), index=False)
    print(f"  wrote well_heads_CGWB.csv ({len(cgwb)} rows) and "
          f"well_heads_daily.csv ({len(daily)} rows)")

    print("\nDone. Run the pipeline against this folder, e.g.:")
    print(f"  python training/optuna_tdr_rar_pinn_training.py --workspace \"{outdir}\"")


if __name__ == "__main__":
    main()
