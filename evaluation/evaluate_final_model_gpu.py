import os
import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import pyproj
import flopy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from PIL import Image
import sys
sys.path.append(r'C:\Users\Shreyansh\.gemini\antigravity\brain\4c84105f-6891-40f1-92e8-beb11d6d579c\scratch')

plt.rcParams['font.family'] = 'Times New Roman'

# Force CPU execution to prevent memory locks with other training runs
os.environ["CUDA_VISIBLE_DEVICES"] = ""

workspace = r'E:\mayank\varuna_refinement'
model_dir = os.path.join(workspace, 'GM47_b1_MODFLOW')
h5_path = os.path.join(model_dir, 'GM47_b1.h5')
chk_w_path = os.path.join(workspace, f'checkpoint_weights_cycle_9_saved.h5')

if not os.path.exists(chk_w_path):
    # fallback to CPU weights if generated on CPU
    chk_w_path = os.path.join(workspace, f'checkpoint_weights_cycle_9_cpu.h5')
    print(f"Warning: Final model weights not found at '{chk_w_path}'.")
    print("If training is still running, you can use checkpoint_weights_cycle_<N>_saved.h5 instead.")

# ==========================================
# 1. Load Grid & MODFLOW Properties
# ==========================================
print("Loading grid properties from H5 file...", flush=True)
with h5py.File(h5_path, 'r') as f:
    top1 = f['Arrays/top1'][:].reshape((105, 217))
    HK = [f[f'Arrays/HK{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    SS = [f[f'Arrays/SS{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    bot = [f[f'Arrays/bot{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    ibound = [f[f'Arrays/ibound{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    VANI = [f[f'Arrays/VANI{l}'][:].reshape((105, 217)) for l in range(1, 8)]

ibound_top = ibound[0]
delr = 747.87628161225
delc = 747.98335290138

with h5py.File(h5_path, 'r') as f:
    chd_cell_ids = f['Specified Head/02. Cell IDs'][:]
    chd_prop = f['Specified Head/07. Property'][:, :, :]
    wel_cell_ids = f['Well/02. Cell IDs'][:]
    str_cell_ids = f['Stream/02. Cell IDs'][:]
    str_prop = f['Stream/07. Property'][:, :, :]

c_chd_grid_3d = np.zeros((7, 105, 217), dtype=np.float32)
h_chd_grid_3d = np.zeros((21, 7, 105, 217), dtype=np.float32)
for idx, cid in enumerate(chd_cell_ids):
    node = cid - 1
    k, r, c = node // (105 * 217), (node % (105 * 217)) // 217, (node % (105 * 217)) % 217
    c_chd_grid_3d[k, r, c] = 1.0
    for sp in range(21):
        h_chd_grid_3d[sp, k, r, c] = chd_prop[0, idx, sp]

c_str_grid = np.zeros((105, 217), dtype=np.float32)
for idx, cid in enumerate(str_cell_ids):
    node = cid - 1
    k, r, c = node // (105 * 217), (node % (105 * 217)) // 217, (node % (105 * 217)) % 217
    if k == 0:
        c_str_grid[r, c] += str_prop[1, idx, 0]

# CHD distances for alpha factor
L_scale = 10000.0
alpha_chd_grid = []
for l in range(7):
    chd_r, chd_c = np.where(c_chd_grid_3d[l] > 0)
    if len(chd_r) > 0:
        chd_x = (chd_c + 0.5) * delr
        chd_y = (chd_r + 0.5) * delc
        grid_c = np.arange(217)
        grid_r = np.arange(105)
        grid_x = ((grid_c + 0.5) * delr).reshape(1, 217)
        grid_y = ((grid_r + 0.5) * delc).reshape(105, 1)
        
        min_dist = np.ones((105, 217), dtype=np.float32) * 1e9
        for cx, cy in zip(chd_x, chd_y):
            dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
            min_dist = np.minimum(min_dist, dist)
        alpha = 1.0 - np.exp(-min_dist / L_scale)
    else:
        alpha = np.ones((105, 217), dtype=np.float32)
    alpha_chd_grid.append(alpha.astype(np.float32))

print("Loading MODFLOW binary heads...", flush=True)
hed_path = os.path.join(model_dir, 'GM47_b1.hed')
hds = flopy.utils.HeadFile(hed_path, precision='double')
head_data = hds.get_alldata() # shape: (110, 7, 105, 217)
times = np.array(hds.get_times())
hds.close()

# Scaler bounds configuration
scaler = {
    'x_l': 0.0, 'x_diff': 217 * delr,
    'y_l': 0.0, 'y_diff': 105 * delc,
    't_l': 0.0, 't_diff': 638.0,
    'hl': np.min(head_data[head_data > -100]), 'hu': np.max(head_data),
}
scaler['hdiff'] = scaler['hu'] - scaler['hl'] + 1e-6

perlen = [30.0, 31.0, 31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 29.0]
cumsum_perlen = np.cumsum(perlen)
def get_sp(t):
    return np.clip(np.searchsorted(cumsum_perlen, t, side='right'), 0, 20)

transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32644', always_xy=True)
X0, Y0, det = 573624.966, 2880613.56, 559399.08
ref_date = pd.to_datetime('2021-10-31')

def evaluate_obs_file_raw(file_path):
    df_obs = pd.read_csv(file_path).dropna(subset=['Ground Water Level (m)'])
    df_obs = df_obs[(df_obs['Ground Water Level (m)'] >= 0.0) & (df_obs['Ground Water Level (m)'] <= 120.0)]
    xs, ys = transformer.transform(df_obs['Long'].values, df_obs['Lat'].values)
    df_obs['grid_row'] = np.round((-225.254385 * (xs - X0) - 713.147527 * (ys - Y0)) / det).astype(int)
    df_obs['grid_col'] = np.round((713.249611 * (xs - X0) - 225.286629 * (ys - Y0)) / det).astype(int)
    df_obs['days'] = (pd.to_datetime(df_obs['DateTime']) - ref_date).dt.days.astype(float)
    valid = (df_obs['grid_row'] >= 0) & (df_obs['grid_row'] < 105) & \
            (df_obs['grid_col'] >= 0) & (df_obs['grid_col'] < 217) & \
            (df_obs['days'] >= 0.0) & (df_obs['days'] <= 638.0)
    df_obs = df_obs[valid]
    df_obs = df_obs[ibound_top[df_obs['grid_row'].values, df_obs['grid_col'].values] > 0]
    return df_obs

print("Loading observation datasets...", flush=True)
cgwb_raw = evaluate_obs_file_raw(os.path.join(workspace, 'well_heads_CGWB.csv'))
daily_raw = evaluate_obs_file_raw(os.path.join(workspace, 'well_heads_daily.csv'))
combined_raw = pd.concat([cgwb_raw, daily_raw], ignore_index=True)

well_mask_eval = np.zeros((105, 217), dtype=bool)
for cid in wel_cell_ids:
    node = cid - 1
    well_mask_eval[(node % (105 * 217)) // 217, (node % (105 * 217)) % 217] = True
river_mask_eval = (c_str_grid > 0)
in_near_eval = river_mask_eval[combined_raw['grid_row'].values, combined_raw['grid_col'].values] | \
               well_mask_eval[combined_raw['grid_row'].values, combined_raw['grid_col'].values]
near_raw = combined_raw[in_near_eval].copy()

times_np = np.array(times)
def get_modflow_heads_at_points(df):
    rows = df['grid_row'].values.astype(int)
    cols = df['grid_col'].values.astype(int)
    days = df['days'].values
    heads = []
    for r, c, d in zip(rows, cols, days):
        h_ts = head_data[:, 0, r, c]
        heads.append(float(np.interp(d, times_np, h_ts)))
    return np.array(heads)

cgwb_raw['H_modflow'] = get_modflow_heads_at_points(cgwb_raw)
daily_raw['H_modflow'] = get_modflow_heads_at_points(daily_raw)
combined_raw['H_modflow'] = get_modflow_heads_at_points(combined_raw)
near_raw['H_modflow'] = get_modflow_heads_at_points(near_raw)

# ==========================================
# 2. Recreate SciANN Model Architecture
# ==========================================
print("Compiling model structure...", flush=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_checkpoint_9_cpu import build_sciann_model, best_params

m, h_net, *_ = build_sciann_model(
    best_params['pde_gw_mult'], best_params['pde_str_mult'], best_params['pde_chd_mult'], best_params['pde_bwell_mult'],
    num_layers=best_params['num_layers'],
    layer_width=best_params['layer_width'],
    num_freqs=best_params['num_freqs'],
    activation=best_params['activation'],
    rff_sigma=best_params['rff_sigma']
)

# Allow custom weights path from command line
if len(sys.argv) > 1:
    target_weights = sys.argv[1]
else:
    target_weights = chk_w_path

print(f"Loading model weights from '{target_weights}'...", flush=True)
m.model.load_weights(target_weights)

def get_pinn_heads_at_points(df_pts):
    x_val = ((df_pts['grid_col'].values + 0.5) * delr).astype(np.float32)
    y_val = ((df_pts['grid_row'].values + 0.5) * delc).astype(np.float32)
    t_val = df_pts['days'].values.astype(np.float32)
    
    x_s = (x_val - scaler['x_l']) / scaler['x_diff']
    y_s = (y_val - scaler['y_l']) / scaler['y_diff']
    t_s = (t_val - scaler['t_l']) / scaler['t_diff']
    
    zeros = np.zeros_like(x_s)
    inp_eval = [x_s.reshape(-1, 1), y_s.reshape(-1, 1), t_s.reshape(-1, 1)]
    for _ in range(75):
        inp_eval.append(zeros.reshape(-1, 1))
        
    pred_scaled = h_net[0].eval(m, inp_eval).flatten()
    pred_phys_raw = pred_scaled * scaler['hdiff'] + scaler['hl']
    
    r = df_pts['grid_row'].values.astype(int)
    c = df_pts['grid_col'].values.astype(int)
    days = df_pts['days'].values
    sp = get_sp(days)
    
    alpha = alpha_chd_grid[0][r, c]
    h_chd = h_chd_grid_3d[sp, 0, r, c]
    
    pred_phys = (1.0 - alpha) * h_chd + alpha * pred_phys_raw
    return pred_phys

def evaluate_obs_file(df_raw):
    df_eval = df_raw.copy()
    df_eval['H_surrogate'] = get_pinn_heads_at_points(df_eval)
    return df_eval

print("Evaluating observations...", flush=True)
cgwb_eval = evaluate_obs_file(cgwb_raw)
daily_eval = evaluate_obs_file(daily_raw)
combined_eval = pd.concat([cgwb_eval, daily_eval], ignore_index=True)

# 4th Dataset: Near Wells & Rivers
in_near_eval = river_mask_eval[combined_eval['grid_row'].values, combined_eval['grid_col'].values] | \
               well_mask_eval[combined_eval['grid_row'].values, combined_eval['grid_col'].values]
near_eval = combined_eval[in_near_eval].copy()

def compute_metrics(df):
    y_obs = df['Ground Water Level (m)'].values
    y_mf = df['H_modflow'].values
    y_pinn = df['H_surrogate'].values
    return {
        'MODFLOW R2': float(r2_score(y_obs, y_mf)),
        'MODFLOW RMSE': float(np.sqrt(mean_squared_error(y_obs, y_mf))),
        'MODFLOW MAE': float(mean_absolute_error(y_obs, y_mf)),
        'PINN R2': float(r2_score(y_obs, y_pinn)),
        'PINN RMSE': float(np.sqrt(mean_squared_error(y_obs, y_pinn))),
        'PINN MAE': float(mean_absolute_error(y_obs, y_pinn))
    }

cgwb_metrics = compute_metrics(cgwb_eval)
daily_metrics = compute_metrics(daily_eval)
combined_metrics = compute_metrics(combined_eval)
near_metrics = compute_metrics(near_eval)

print("\n" + "="*60, flush=True)
print("EVALUATION METRICS FOR FINAL LOADED MODEL", flush=True)
print("="*60, flush=True)
for name, metrics in [('CGWB', cgwb_metrics), ('Daily', daily_metrics), ('Combined', combined_metrics), ('Near Wells/Rivers', near_metrics)]:
    print(f"\n{name} Dataset ({len(cgwb_eval) if name=='CGWB' else len(daily_eval) if name=='Daily' else len(combined_eval) if name=='Combined' else len(near_eval)} observations):", flush=True)
    print(f"  MODFLOW: R2={metrics['MODFLOW R2']:.4f}, RMSE={metrics['MODFLOW RMSE']:.4f} m, MAE={metrics['MODFLOW MAE']:.4f} m", flush=True)
    print(f"  PINN:    R2={metrics['PINN R2']:.4f}, RMSE={metrics['PINN RMSE']:.4f} m, MAE={metrics['PINN MAE']:.4f} m", flush=True)
print("="*60 + "\n", flush=True)

# Export Evaluation CSVs
cgwb_eval.to_csv(os.path.join(workspace, 'cgwb_evaluation_results.csv'), index=False)
daily_eval.to_csv(os.path.join(workspace, 'daily_evaluation_results.csv'), index=False)
combined_eval.to_csv(os.path.join(workspace, 'combined_evaluation_results.csv'), index=False)
near_eval.to_csv(os.path.join(workspace, 'near_evaluation_results.csv'), index=False)
print("Saved evaluation CSVs successfully!", flush=True)

# Save scatter plots
def save_scatter_plot(df, title, path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df['Ground Water Level (m)'], df['H_modflow'], alpha=0.6, color='blue', label='MODFLOW')
    plt.plot([df['Ground Water Level (m)'].min(), df['Ground Water Level (m)'].max()],
             [df['Ground Water Level (m)'].min(), df['Ground Water Level (m)'].max()], 'r--')
    plt.title(f'MODFLOW vs Observed ({title})')
    plt.xlabel('Observed Head (m)')
    plt.ylabel('Simulated Head (m)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['Ground Water Level (m)'], df['H_surrogate'], alpha=0.6, color='green', label='PINN')
    plt.plot([df['Ground Water Level (m)'].min(), df['Ground Water Level (m)'].max()],
             [df['Ground Water Level (m)'].min(), df['Ground Water Level (m)'].max()], 'r--')
    plt.title(f'PINN vs Observed ({title})')
    plt.xlabel('Observed Head (m)')
    plt.ylabel('Predicted Head (m)')
    plt.grid(True)
    plt.tight_layout()
    base_path = path.rsplit('.', 1)[0]
    plt.savefig(base_path + '.svg', format='svg', dpi=300)
    plt.savefig(base_path + '.png', format='png', dpi=300)
    plt.savefig(base_path + '.tif', format='tiff', dpi=300)
    plt.close()

save_scatter_plot(cgwb_eval, 'CGWB', os.path.join(workspace, 'cgwb_comparison.svg'))

daily_mae_mf = np.abs(daily_eval['Ground Water Level (m)'] - daily_eval['H_modflow'])
daily_eval_filtered = daily_eval[daily_mae_mf <= 25].copy()
save_scatter_plot(daily_eval_filtered, 'Daily', os.path.join(workspace, 'daily_comparison.svg'))

save_scatter_plot(near_eval, 'Near Wells & Rivers', os.path.join(workspace, 'near_comparison.svg'))

# ==========================================
# 3. Spatial mid-timestep Comparison Heatmap
# ==========================================
print("Generating mid-timestep comparison heatmap...", flush=True)
mid_t_idx = len(times) // 2
t_mid = times[mid_t_idx]
h_mf_mid = head_data[mid_t_idx, 0, :, :].copy()

r_grid, c_grid = np.meshgrid(np.arange(105), np.arange(217), indexing='ij')
x_grid = ((c_grid + 0.5) * delr).astype(np.float32)
y_grid = ((r_grid + 0.5) * delc).astype(np.float32)
t_grid = np.full_like(x_grid, t_mid, dtype=np.float32)

x_s = (x_grid - scaler['x_l']) / scaler['x_diff']
y_s = (y_grid - scaler['y_l']) / scaler['y_diff']
t_s = (t_grid - scaler['t_l']) / scaler['t_diff']

zeros_plot = np.zeros_like(x_s)
inp_plot = [x_s.reshape(-1, 1), y_s.reshape(-1, 1), t_s.reshape(-1, 1)]
for _ in range(75):
    inp_plot.append(zeros_plot.reshape(-1, 1))

pred_scaled = h_net[0].eval(m, inp_plot).reshape((105, 217))
pred_phys_raw = pred_scaled * scaler['hdiff'] + scaler['hl']

sp_mid = get_sp(t_mid)
alpha_mid = alpha_chd_grid[0]
h_chd_mid = h_chd_grid_3d[sp_mid, 0, :, :]
h_pinn_mid = (1.0 - alpha_mid) * h_chd_mid + alpha_mid * pred_phys_raw

h_mf_mid[ibound_top <= 0] = np.nan
h_pinn_mid[ibound_top <= 0] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
vmin_mf = np.nanpercentile(h_mf_mid, 1)
vmax_mf = np.nanpercentile(h_mf_mid, 99)
vmin_pinn = np.nanpercentile(h_pinn_mid, 5) # 5th percentile filters outliers
vmax_pinn = np.nanpercentile(h_pinn_mid, 95) # 95th percentile filters outliers

im1 = axes[0].imshow(h_mf_mid, cmap='viridis', vmin=vmin_mf, vmax=vmax_mf)
axes[0].set_title(f'MODFLOW Head Variation at Mid-Timestep (t={t_mid:.1f} d)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Col Index')
axes[0].set_ylabel('Row Index')
fig.colorbar(im1, ax=axes[0], label='Head (m)', shrink=0.75)

im2 = axes[1].imshow(h_pinn_mid, cmap='viridis', vmin=vmin_pinn, vmax=vmax_pinn)
axes[1].set_title(f'PINN Head Prediction at Mid-Timestep (t={t_mid:.1f} d)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Col Index')
axes[1].set_ylabel('Row Index')
fig.colorbar(im2, ax=axes[1], label='Head (m)', shrink=0.75)

plt.tight_layout()
plt.savefig(os.path.join(workspace, 'mid_timestep_comparison.svg'), format='svg', dpi=300)
plt.savefig(os.path.join(workspace, 'mid_timestep_comparison.tif'), format='tiff', dpi=300)
plt.close()

# Export raw grids as TIFF
h_mf_mid_out = h_mf_mid.copy()
h_pinn_mid_out = h_pinn_mid.copy()
h_mf_mid_out[np.isnan(h_mf_mid_out)] = -999.0
h_pinn_mid_out[np.isnan(h_pinn_mid_out)] = -999.0
Image.fromarray(h_mf_mid_out.astype(np.float32)).save(os.path.join(workspace, 'modflow_mid_head.tif'))
Image.fromarray(h_pinn_mid_out.astype(np.float32)).save(os.path.join(workspace, 'pinn_mid_head.tif'))

# ==========================================
# 4. Spatial MAE Heatmap
# ==========================================
print("Generating spatial MAE heatmap for daily observations...", flush=True)
daily_eval['Err_MF'] = np.abs(daily_eval['H_modflow'] - daily_eval['Ground Water Level (m)'])
daily_eval['Err_PINN'] = np.abs(daily_eval['H_surrogate'] - daily_eval['Ground Water Level (m)'])

daily_well_stats = daily_eval.groupby(['grid_row', 'grid_col']).agg({
    'Err_MF': 'mean',
    'Err_PINN': 'mean',
    'DateTime': 'count'
}).reset_index().rename(columns={'DateTime': 'obs_count'})

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
max_err_to_show = 15.0
vmin, vmax = 0.0, max_err_to_show

for idx, (ax, col_err, title) in enumerate([
    (axes[0], 'Err_MF', 'MODFLOW MAE Spatial Distribution (vs. Daily Observations)'),
    (axes[1], 'Err_PINN', 'PINN MAE Spatial Distribution (vs. Daily Observations)')
]):
    aquifer_mask = np.zeros_like(ibound_top, dtype=float)
    aquifer_mask[ibound_top > 0] = 0.9
    aquifer_mask[ibound_top <= 0] = 0.2
    
    ax.imshow(aquifer_mask, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0.3)
    
    sc = ax.scatter(
        daily_well_stats['grid_col'], daily_well_stats['grid_row'],
        c=daily_well_stats[col_err], cmap='Reds', vmin=vmin, vmax=vmax,
        s=35, edgecolors='black', linewidths=0.5, alpha=0.9
    )
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Grid Column Index', fontsize=11)
    ax.set_ylabel('Grid Row Index', fontsize=11)
    ax.set_xlim(-2, 218)
    ax.set_ylim(-2, 106)
    ax.invert_yaxis()
    
    cbar = fig.colorbar(sc, ax=ax, label='Mean Absolute Error (meters)', extend='max', shrink=0.75)
    cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(workspace, 'spatial_error_comparison_daily.svg'), format='svg', dpi=300)
plt.close()
print("All final plots and evaluation CSVs generated successfully!", flush=True)
