import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pyproj

workspace = r'E:\mayank\varuna_refinement\try_1_master'
drop_dir = os.path.join(workspace, 'drop_1000')

# 1. Load active cells (ibound) for the background
h5_path = os.path.join(workspace, 'GM47_b1_MODFLOW', 'GM47_b1.h5')
with h5py.File(h5_path, 'r') as f:
    ibound_top = f['Arrays/ibound1'][:].reshape((105, 217))

aquifer_mask = np.zeros_like(ibound_top, dtype=float)
aquifer_mask[ibound_top > 0] = 0.95
aquifer_mask[ibound_top <= 0] = 0.8

# 2. Extract unique field observation locations
obs_csv = os.path.join(drop_dir, 'daily_dataset_modflow_drop1000.csv')
df_obs = pd.read_csv(obs_csv)
unique_wells = df_obs[['Lat', 'Long', 'WellID']].drop_duplicates()

# Proj transformer: EPSG:4326 (WGS84 Lat/Long) to EPSG:32644 (UTM 44N)
# Note: pyproj Transformer expects (lat, lon) or (lon, lat) depending on always_xy
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32644", always_xy=True)
xs, ys = transformer.transform(unique_wells['Long'].values, unique_wells['Lat'].values)

# Grid origin and transform details
X0, Y0 = 573624.966, 2880613.56
det = 559399.08

# Convert UTM to Grid Row and Column
grid_row = np.round((-225.254385 * (xs - X0) - 713.147527 * (ys - Y0)) / det).astype(int)
grid_col = np.round((713.249611 * (xs - X0) - 225.286629 * (ys - Y0)) / det).astype(int)

# 3. Load RAR points
points_csv = os.path.join(workspace, 'other files', 'rar_points_all.csv')
df_pts = pd.read_csv(points_csv)

# 4. Plotting 3x3 grid
plt.style.use('default')
fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
axes = axes.flatten()

cycles_to_plot = sorted(df_pts['cycle'].unique())
if len(cycles_to_plot) > 9:
    cycles_to_plot = cycles_to_plot[:9]

for i, ax in enumerate(axes):
    if i < len(cycles_to_plot):
        cycle_num = cycles_to_plot[i]
        
        # Background
        ax.imshow(aquifer_mask, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0.3)
        
        # Base Field Observations (Ghosted for context, no legend)
        ax.scatter(
            grid_col, grid_row,
            color='white', marker='*', s=60, alpha=0.4, edgecolors='gray', linewidth=0.5
        )
        
        # PDE Points (Modflow Domain)
        sub_pts_pde = df_pts[(df_pts['cycle'] == cycle_num) & (df_pts['type'] == 'PDE')]
        if len(sub_pts_pde) > 0:
            j_col = sub_pts_pde['col'] + np.random.uniform(-0.25, 0.25, size=len(sub_pts_pde))
            j_row = sub_pts_pde['row'] + np.random.uniform(-0.25, 0.25, size=len(sub_pts_pde))
            scatter_pde = ax.scatter(
                j_col, j_row,
                color='#1f77b4', label=f'Modflow Model Points',
                s=12, alpha=0.6, edgecolors='none'
            )
            
        # MAE Points (Field Obs)
        sub_pts_mae = df_pts[(df_pts['cycle'] == cycle_num) & (df_pts['type'] == 'MAE')]
        if len(sub_pts_mae) > 0:
            j_col = sub_pts_mae['col'] + np.random.uniform(-0.25, 0.25, size=len(sub_pts_mae))
            j_row = sub_pts_mae['row'] + np.random.uniform(-0.25, 0.25, size=len(sub_pts_mae))
            scatter_mae = ax.scatter(
                j_col, j_row,
                color='#d62728', marker='s', label=f'Field Observation Points',
                s=12, alpha=0.8, edgecolors='none'
            )
        
        ax.set_title(f'Cycle {cycle_num}', fontsize=12, fontweight='bold', pad=8)
        ax.set_xlim(-2, 218)
        ax.set_ylim(-2, 106)
        ax.invert_yaxis()
        
        if i % 3 == 0:
            ax.set_ylabel('Grid Row Index', fontsize=10)
        else:
            ax.set_yticks([])
            
        if i >= 6:
            ax.set_xlabel('Grid Column Index', fontsize=10)
        else:
            ax.set_xticks([])
            
    else:
        ax.axis('off')

# Common Legend at the bottom
handles = [scatter_mae, scatter_pde]
fig.legend(handles, [h.get_label() for h in handles], loc='lower center', ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.02))

fig.suptitle('Spatial Evolution of High Error TDR-RAR Collocation Points (drop_1000)', fontsize=16, fontweight='bold', y=0.96)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])

out_dir = os.path.join(drop_dir, 'plots')
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, 'rar_spatial_distribution_3x3.png')
out_svg = os.path.join(out_dir, 'rar_spatial_distribution_3x3.svg')

plt.savefig(out_png, format='png', dpi=300)
plt.savefig(out_svg, format='svg', dpi=300)
plt.close()

print(f"3x3 plot saved successfully at {out_png}")
