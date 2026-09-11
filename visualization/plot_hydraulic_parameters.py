import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['font.family'] = 'Times New Roman'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plots_dir = os.path.join(workspace, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

model_dir = os.path.join(workspace, 'GM47_b1_MODFLOW')
h5_path = os.path.join(model_dir, 'GM47_b1.h5')

print("Loading grid properties, HK and SS data...")
with h5py.File(h5_path, 'r') as f:
    ibound = [f[f'Arrays/ibound{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    HK_mf = [f[f'Arrays/HK{l}'][:].reshape((105, 217)).astype(float) for l in range(1, 8)]
    SS_mf = [f[f'Arrays/SS{l}'][:].reshape((105, 217)).astype(float) for l in range(1, 8)]

for l in range(7):
    HK_mf[l][ibound[l] <= 0] = np.nan
    SS_mf[l][ibound[l] <= 0] = np.nan

print("Evaluating PINN K and S...")
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_checkpoint_9_cpu import build_sciann_model, best_params, scaler

m, h_net, param_net_logK, param_net_S, *_ = build_sciann_model(
    best_params['pde_gw_mult'], best_params['pde_str_mult'], best_params['pde_chd_mult'], best_params['pde_bwell_mult'],
    num_layers=best_params['num_layers'],
    layer_width=best_params['layer_width'],
    num_freqs=best_params['num_freqs'],
    activation=best_params['activation'],
    rff_sigma=best_params['rff_sigma']
)

chk_w_path = os.path.join(workspace, f'checkpoint_weights_cycle_9_saved.h5')
if not os.path.exists(chk_w_path):
    chk_w_path = os.path.join(workspace, f'checkpoint_weights_cycle_9_cpu.h5')
m.model.load_weights(chk_w_path)

delr, delc = 747.87628161225, 747.98335290138
cols, rows = np.meshgrid(np.arange(217), np.arange(105))
x_val = ((cols + 0.5) * delr).astype(np.float32).flatten()
y_val = ((rows + 0.5) * delc).astype(np.float32).flatten()
x_s = (x_val - scaler['x_l']) / scaler['x_diff']
y_s = (y_val - scaler['y_l']) / scaler['y_diff']

def plot_parameter_heatmap_log(mf_arr, pinn_arr, title, filename, cmap='viridis'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    all_vals = np.concatenate([mf_arr[~np.isnan(mf_arr)], pinn_arr[~np.isnan(pinn_arr)]])
    all_vals = all_vals[all_vals > 0]
    vmin = max(np.nanpercentile(all_vals, 2), 1e-10)
    vmax = np.nanpercentile(all_vals, 98)
    
    norm = LogNorm(vmin=vmin, vmax=vmax)
    
    for c_idx, (arr, subtit) in enumerate(zip([mf_arr, pinn_arr], ["MODFLOW", "PINN"])):
        ax = axes[c_idx]
        im = ax.imshow(arr, cmap=cmap, norm=norm)
        ax.set_title(f"{subtit} {title}", fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        
        x_ticks = np.linspace(0, 217, 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x * delr:.0f}" for x in x_ticks])
        
        y_ticks = np.linspace(0, 105, 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y * delc:.0f}" for y in y_ticks])
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, label=title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{filename}.png'), dpi=300)
    plt.savefig(os.path.join(plots_dir, f'{filename}.svg'), format='svg', dpi=300)
    plt.savefig(os.path.join(plots_dir, f'{filename}.tif'), format='tiff', dpi=300)
    plt.close()

for l in range(7):
    print(f"Processing layer {l}...")
    # Hydraulic Conductivity (K)
    logK_pred = param_net_logK[l].eval([x_s.reshape(-1,1), y_s.reshape(-1,1)]).flatten()
    K_phys = np.exp((logK_pred * scaler['K_log_diff'] + scaler['K_log_l']) * np.log(10.0))
    hk_pinn = K_phys.reshape(105, 217)
    hk_pinn[ibound[l] <= 0] = np.nan
    
    plot_parameter_heatmap_log(HK_mf[l], hk_pinn, f"Hydraulic Conductivity - Layer {l} (m/d)", f"k_comparison_heatmap_log_layer_{l}")

    # Specific Storage (S)
    S_pred = param_net_S[l].eval([x_s.reshape(-1,1), y_s.reshape(-1,1)]).flatten()
    S_phys = S_pred * scaler['S_diff'] + scaler['S_l']
    ss_pinn = S_phys.reshape(105, 217)
    ss_pinn[ibound[l] <= 0] = np.nan
    
    plot_parameter_heatmap_log(SS_mf[l], ss_pinn, f"Specific Storage - Layer {l} (1/m)", f"s_comparison_heatmap_log_layer_{l}")

print("All K and S layers generated successfully!")
