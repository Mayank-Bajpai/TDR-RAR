import os
# Force TensorFlow to run on CPU to avoid device lock/contention with the active GPU training process
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import sciann as sn
from sciann.utils.math import diff, relu, sign, exp
import pyproj
import flopy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from PIL import Image

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(workspace, 'GM47_b1_MODFLOW')
h5_path = os.path.join(model_dir, 'GM47_b1.h5')

# ==========================================
# 1. Load Grid & MODFLOW Properties
# ==========================================
print("Loading grid properties from H5 file...", flush=True)
with h5py.File(h5_path, 'r') as f:
    top1 = f['Arrays/top1'][:].reshape((105, 217))
    HK = [f[f'Arrays/HK{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    SS = [f[f'Arrays/SS{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    SY = [f[f'Arrays/SY{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    bot = [f[f'Arrays/bot{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    ibound = [f[f'Arrays/ibound{l}'][:].reshape((105, 217)) for l in range(1, 8)]
    VANI = [f[f'Arrays/VANI{l}'][:].reshape((105, 217)) for l in range(1, 8)]

ibound_top = ibound[0]
delr = 747.87628161225
delc = 747.98335290138

top_layers = [top1] + [bot[l-1] for l in range(1, 7)]
b_layers = [top_layers[l] - bot[l] for l in range(7)]
vol_layers = [(b_layers[l] * delr * delc).astype(np.float32) for l in range(7)]

with h5py.File(h5_path, 'r') as f:
    chd_cell_ids = f['Specified Head/02. Cell IDs'][:]
    chd_prop = f['Specified Head/07. Property'][:, :, :]
    wel_cell_ids = f['Well/02. Cell IDs'][:]
    wel_prop = f['Well/07. Property'][:, :, :]
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

Q_wells_grid_3d = np.zeros((21, 7, 105, 217), dtype=np.float32)
for idx, cid in enumerate(wel_cell_ids):
    node = cid - 1
    k, r, c = node // (105 * 217), (node % (105 * 217)) // 217, (node % (105 * 217)) % 217
    for sp in range(21):
        Q_wells_grid_3d[sp, k, r, c] += wel_prop[0, idx, sp]

c_str_grid = np.zeros((105, 217), dtype=np.float32)
h_bot_grid = np.zeros((105, 217), dtype=np.float32)
for idx, cid in enumerate(str_cell_ids):
    node = cid - 1
    k, r, c = node // (105 * 217), (node % (105 * 217)) // 217, (node % (105 * 217)) % 217
    if k == 0:
        c_str_grid[r, c] += str_prop[1, idx, 0]
        if h_bot_grid[r, c] == 0.0:
            h_bot_grid[r, c] = str_prop[2, idx, 0]

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

# Load MODFLOW simulated heads and stress period times
print("Loading MODFLOW binary outputs...", flush=True)
hed_path = os.path.join(model_dir, 'GM47_b1.hed')
hds = flopy.utils.HeadFile(hed_path, precision='double')
head_data = hds.get_alldata() # shape: (110, 7, 105, 217)
times = np.array(hds.get_times())
hds.close()

cbc_path = os.path.join(model_dir, 'GM47_b1.ccf')
cbc = flopy.utils.CellBudgetFile(cbc_path, precision='double')
rch_records = cbc.get_data(text='RECHARGE')
et_records = cbc.get_data(text='ET')
str_records = cbc.get_data(text='STREAM LEAKAGE')
cbc.close()

Q_bg_grid = np.zeros((110, 105, 217), dtype=np.float32)
Q_leak_grid = np.zeros((110, 105, 217), dtype=np.float32)
for t_idx in range(110):
    Q_bg_grid[t_idx, :, :] = rch_records[t_idx][0] + et_records[t_idx][0]
    for r in str_records[t_idx]:
        node = r['node'] - 1
        k, rem = node // (105 * 217), node % (105 * 217)
        if k == 0:
            Q_leak_grid[t_idx, rem // 217, rem % 217] += r['q']

h_stage_ts = np.zeros((110, 105, 217), dtype=np.float32)
for t_idx in range(110):
    h_target_step = head_data[t_idx, 0, :, :]
    h_for_leak = np.maximum(h_target_step, h_bot_grid)
    river_mask = (c_str_grid > 0)
    h_stage_ts[t_idx, river_mask] = (Q_leak_grid[t_idx, river_mask] / c_str_grid[river_mask]) + h_for_leak[river_mask]

perlen = [30.0, 31.0, 31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 29.0]
cumsum_perlen = np.cumsum(perlen)
def get_sp(t):
    return np.clip(np.searchsorted(cumsum_perlen, t, side='right'), 0, 20)

HK_active = np.concatenate([HK[l][ibound[l] > 0] for l in range(7)])
HK_active = HK_active[HK_active > 0]
SS_active = np.concatenate([SS[l][ibound[l] > 0] for l in range(7)])
SS_active = SS_active[SS_active > 0]
vol_active = np.concatenate([vol_layers[l][ibound[l] > 0] for l in range(7)])
thick_active = np.concatenate([b_layers[l][ibound[l] > 0] for l in range(7)])

scaler = {
    'x_l': 0.0, 'x_diff': 217 * delr,
    'y_l': 0.0, 'y_diff': 105 * delc,
    't_l': 0.0, 't_diff': 638.0,
    'K_log_l': np.min(np.log10(HK_active)), 'K_log_diff': np.max(np.log10(HK_active)) - np.min(np.log10(HK_active)) + 1e-6,
    'S_l': np.min(SS_active), 'S_diff': np.max(SS_active) - np.min(SS_active) + 1e-6,
    'Vol_l': np.min(vol_active), 'Vol_diff': np.max(vol_active) - np.min(vol_active) + 1e-6,
    'Thick_l': np.min(thick_active), 'Thick_diff': np.max(thick_active) - np.min(thick_active) + 1e-6,
    'Q_bg_l': np.min(Q_bg_grid), 'Q_bg_diff': np.max(Q_bg_grid) - np.min(Q_bg_grid) + 1e-6,
    'Q_wells_l': np.min(Q_wells_grid_3d), 'Q_wells_diff': np.max(Q_wells_grid_3d) - np.min(Q_wells_grid_3d) + 1e-6,
    'C_str_l': np.min(c_str_grid), 'C_str_diff': np.max(c_str_grid) - np.min(c_str_grid) + 1e-6,
    'H_stage_l': np.min(h_stage_ts), 'H_stage_diff': np.max(h_stage_ts) - np.min(h_stage_ts) + 1e-6,
    'H_bot_l': np.min(h_bot_grid), 'H_bot_diff': np.max(h_bot_grid) - np.min(h_bot_grid) + 1e-6,
    'Q_leak_l': np.min(Q_leak_grid), 'Q_leak_diff': np.max(Q_leak_grid) - np.min(Q_leak_grid) + 1e-6,
    'hl': np.min(head_data[head_data > -100]), 'hu': np.max(head_data),
}
scaler['hdiff'] = scaler['hu'] - scaler['hl'] + 1e-6

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

# Fast interpolator using np.interp
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

# ==========================================
# 2. Define SciANN Network Builder Function (Self-Contained)
# ==========================================
def build_sciann_model(pde_gw_mult, pde_str_mult, pde_chd_mult, pde_bwell_mult, num_layers=4, layer_width=256, num_freqs=4, activation='tanh', rff_sigma=2.0, rff_features=None):
    x_var = sn.Variable('x')
    y_var = sn.Variable('y')
    t_var = sn.Variable('t')
    
    K_prior = [sn.Variable(f'K_prior_{l}') for l in range(7)]
    S_prior = [sn.Variable(f'S_prior_{l}') for l in range(7)]
    Vol = [sn.Variable(f'Vol_{l}') for l in range(7)]
    Thickness = [sn.Variable(f'Thickness_{l}') for l in range(7)]
    Q_wells = [sn.Variable(f'Q_wells_{l}') for l in range(7)]
    H_chd = [sn.Variable(f'H_chd_{l}') for l in range(7)]
    alpha_chd = [sn.Variable(f'alpha_chd_{l}') for l in range(7)]
    VANI = [sn.Variable(f'VANI_{l}') for l in range(7)]
    H_head_target = [sn.Variable(f'H_head_target_{l}') for l in range(7)]
    data_mask = [sn.Variable(f'data_mask_{l}') for l in range(7)]
    
    Q_bg_var = sn.Variable('Q_bg')
    C_str_var = sn.Variable('C_str')
    H_stage_var = sn.Variable('H_stage')
    H_bot_var = sn.Variable('H_bot')
    Q_leak_true_var = sn.Variable('Q_leak_true')
    
    inputs = [x_var, y_var, t_var] + K_prior + S_prior + Vol + Thickness + Q_wells + H_chd + alpha_chd + VANI + H_head_target + data_mask + [Q_bg_var, C_str_var, H_stage_var, H_bot_var, Q_leak_true_var]
    
    def get_rff_encoding(x_var, y_var, t_var, num_features=None, sigma=2.0):
        from sciann.utils.math import sin, cos
        if num_features is None:
            num_features = num_freqs * 10
            
        rng = np.random.default_rng(42)
        B_mat = rng.normal(0.0, sigma, (num_features, 3))
        
        encoding = [x_var, y_var, t_var]
        for i in range(num_features):
            b_x, b_y, b_t = B_mat[i]
            proj = b_x * x_var + b_y * y_var + b_t * t_var
            encoding.append(sin(2.0 * np.pi * proj))
            encoding.append(cos(2.0 * np.pi * proj))
        return encoding
        
    pe = get_rff_encoding(x_var, y_var, t_var, num_features=rff_features, sigma=rff_sigma)
    
    # MLP structures
    h_net = sn.Functional([f'h_{l}' for l in range(7)], pe, [layer_width] * num_layers, activation, res_net=True)
    param_net_logK = sn.Functional([f'logK_{l}' for l in range(7)], [x_var, y_var], [128, 128], activation, output_activation='sigmoid')
    param_net_S = sn.Functional([f'S_param_{l}' for l in range(7)], [x_var, y_var], [128, 128], activation, output_activation='sigmoid')
    
    # Hard boundary constraints for CHD
    heads_ansatz = []
    for l in range(7):
        h_ansatz = (1.0 - alpha_chd[l]) * H_chd[l] + alpha_chd[l] * h_net[l]
        heads_ansatz.append(h_ansatz)
        
    # Vertical leakance coupling between layers
    CV = []
    for l in range(6):
        VK_l = exp((param_net_logK[l] * scaler['K_log_diff'] + scaler['K_log_l']) * np.log(10.0)) / (VANI[l] + 1e-9)
        VK_next = exp((param_net_logK[l+1] * scaler['K_log_diff'] + scaler['K_log_l']) * np.log(10.0)) / (VANI[l+1] + 1e-9)
        
        half_thick_l = (Thickness[l] * scaler['Thick_diff'] + scaler['Thick_l']) / 2.0
        half_thick_next = (Thickness[l+1] * scaler['Thick_diff'] + scaler['Thick_l']) / 2.0
        
        cv_l = 1.0 / (half_thick_l / VK_l + half_thick_next / VK_next + 1e-9)
        CV.append(cv_l)
        
    pde_equations = []
    Q_flow_layers = []
    Q_storage_layers = []
    Q_wells_layers = []
    Q_bg_phys_layer0 = 0.0
    Q_leak_phys_layer0 = 0.0
    
    for l in range(7):
        h_l = heads_ansatz[l]
        
        # Hydraulic parameters
        K_phys = exp((param_net_logK[l] * scaler['K_log_diff'] + scaler['K_log_l']) * np.log(10.0))
        S_phys = param_net_S[l] * scaler['S_diff'] + scaler['S_l']
        Vol_phys = Vol[l] * scaler['Vol_diff'] + scaler['Vol_l']
        Thick_phys = Thickness[l] * scaler['Thick_diff'] + scaler['Thick_l']
        
        # Spatial flow
        dh_dx = diff(h_l, x_var)
        dh_dy = diff(h_l, y_var)
        A = K_phys * Thick_phys * (scaler['hdiff'] / scaler['x_diff']) * dh_dx
        B_var_diff = K_phys * Thick_phys * (scaler['hdiff'] / scaler['y_diff']) * dh_dy
        
        h2x = (1.0 / scaler['x_diff']) * diff(A, x_var)
        h2y = (1.0 / scaler['y_diff']) * diff(B_var_diff, y_var)
        Q_flow = Vol_phys * (h2x + h2y)
        Q_flow_layers.append(Q_flow)
        
        # Transient storage
        dh_dt = diff(h_l, t_var) * (scaler['hdiff'] / scaler['t_diff'])
        Q_storage = Vol_phys * S_phys * dh_dt
        Q_storage_layers.append(Q_storage)
        
        # Vertical leakages
        q_z_in = 0.0
        q_z_out = 0.0
        
        if l > 0:
            h_above = heads_ansatz[l-1]
            q_z_in = (delr * delc) * CV[l-1] * ((h_above - h_l) * scaler['hdiff'])
            
        if l < 6:
            h_below = heads_ansatz[l+1]
            q_z_out = (delr * delc) * CV[l] * ((h_l - h_below) * scaler['hdiff'])
            
        # Pumping well rate
        Q_wells_phys = Q_wells[l] * scaler['Q_wells_diff'] + scaler['Q_wells_l']
        Q_wells_layers.append(Q_wells_phys)
        
        # Layer 0 specific terms
        Q_bg_phys = 0.0
        Q_leak_phys = 0.0
        if l == 0:
            Q_bg_phys = Q_bg_var * scaler['Q_bg_diff'] + scaler['Q_bg_l']
            
            h_phys = h_l * scaler['hdiff'] + scaler['hl']
            C_str_phys = C_str_var * scaler['C_str_diff'] + scaler['C_str_l']
            H_stage_phys = H_stage_var * scaler['H_stage_diff'] + scaler['H_stage_l']
            H_bot_phys = H_bot_var * scaler['H_bot_diff'] + scaler['H_bot_l']
            
            head_for_leak = relu(h_phys - H_bot_phys) + H_bot_phys
            Q_leak_phys = C_str_phys * (H_stage_phys - head_for_leak)
            
            Q_bg_phys_layer0 = Q_bg_phys
            Q_leak_phys_layer0 = Q_leak_phys
            
        # Physics residual (unweighted)
        pde_gw = 1e-4 * (Q_flow + q_z_in - q_z_out - Q_storage + Q_bg_phys + Q_wells_phys + Q_leak_phys) * pde_gw_mult
        pde_equations.append(pde_gw)
        
    targets = []
    
    # 1. Domain head matching loss for each layer (using data_mask)
    for l in range(7):
        pde_head_l = data_mask[l] * (heads_ansatz[l] - H_head_target[l])
        pde_head_l._layers[-1]._name = f'h_data_{l}'
        targets.append(sn.Data(pde_head_l))
        
    # 2. PDE flow conservation loss for each layer
    for l in range(7):
        pde_gw_l = pde_equations[l]
        pde_gw_l._layers[-1]._name = f'pde_gw_{l}'
        targets.append(sn.Data(pde_gw_l))
        
    # 3. Stream leakage loss (Layer 0 only)
    Q_leak_phys_scaled = (Q_leak_phys_layer0 - scaler['Q_leak_l']) / scaler['Q_leak_diff']
    pde_str = pde_str_mult * (Q_leak_phys_scaled - Q_leak_true_var)
    pde_str._layers[-1]._name = 'pde_str'
    targets.append(sn.Data(pde_str))
    
    # 4. Boundary well loss for each layer
    for l in range(7):
        is_bwell = sign(Q_wells[l])
        pde_bwell_l = is_bwell * pde_bwell_mult * (heads_ansatz[l] - H_head_target[l])
        pde_bwell_l._layers[-1]._name = f'pde_bwell_{l}'
        targets.append(sn.Data(pde_bwell_l))
        
    # 5. Parameter network regularization losses (with 90% relaxation at observations)
    w_reg = 0.01
    for l in range(7):
        pde_K_reg_l = w_reg * (1.0 - 0.9 * data_mask[l]) * (param_net_logK[l] - K_prior[l])
        pde_K_reg_l._layers[-1]._name = f'pde_K_reg_{l}'
        pde_S_reg_l = w_reg * (1.0 - 0.9 * data_mask[l]) * (param_net_S[l] - S_prior[l])
        pde_S_reg_l._layers[-1]._name = f'pde_S_reg_{l}'
        targets.append(sn.Data(pde_K_reg_l))
        targets.append(sn.Data(pde_S_reg_l))
        
    m = sn.SciModel(inputs, targets, optimizer='adam', loss_func="mse")
    return m, h_net, param_net_logK, param_net_S, Q_flow_layers, Q_storage_layers, Q_wells_layers, Q_leak_phys_layer0, Q_bg_phys_layer0, pde_equations, pde_str

print("Compiling model structure...", flush=True)
best_params = {
    'pde_gw_mult': 1.840870,
    'pde_str_mult': 198.962600,
    'pde_chd_mult': 3.470069,
    'pde_bwell_mult': 263.454971,
    'batch_size': 1024,
    'epochs': 918,
    'lr': 0.001045,
    'num_layers': 6,
    'layer_width': 256,
    'activation': 'gelu',
    'num_freqs': 2,
    'total_pts': 50000,
    'rff_sigma': 2.0
}

m, h_net, *_ = build_sciann_model(
    best_params['pde_gw_mult'], best_params['pde_str_mult'], best_params['pde_chd_mult'], best_params['pde_bwell_mult'],
    num_layers=best_params['num_layers'],
    layer_width=best_params['layer_width'],
    num_freqs=best_params['num_freqs'],
    activation=best_params['activation'],
    rff_sigma=best_params['rff_sigma']
)

# Load weights from Cycle 3 checkpoint
chk_w_path = os.path.join(workspace, "checkpoint_weights_cycle_9_saved.h5")
print(f"Loading weights from: {chk_w_path}", flush=True)
m.model.load_weights(chk_w_path)

# ==========================================
# 3. Evaluation on Observation Wells
# ==========================================
def get_pinn_heads_at_points(df_pts):
    x_val = ((df_pts['grid_col'].values + 0.5) * delr).astype(np.float32)
    y_val = ((df_pts['grid_row'].values + 0.5) * delc).astype(np.float32)
    t_val = df_pts['days'].values.astype(np.float32)
    
    x_s = (x_val - scaler['x_l']) / scaler['x_diff']
    y_s = (y_val - scaler['y_l']) / scaler['y_diff']
    t_s = (t_val - scaler['t_l']) / scaler['t_diff']
    
    zeros = np.zeros_like(x_s)
    inp_eval = [
        x_s.reshape(-1, 1), y_s.reshape(-1, 1), t_s.reshape(-1, 1)
    ]
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
    
    # Correctly apply hard CHD boundary condition ansatz in physical units
    pred_phys = (1.0 - alpha) * h_chd + alpha * pred_phys_raw
    return pred_phys

def evaluate_obs_file(df_raw):
    df_eval = df_raw.copy()
    df_eval['H_modflow'] = get_modflow_heads_at_points(df_eval)
    df_eval['H_surrogate'] = get_pinn_heads_at_points(df_eval)
    return df_eval

print("Evaluating metrics on datasets...", flush=True)
cgwb_eval = evaluate_obs_file(cgwb_raw)
daily_eval = evaluate_obs_file(daily_raw)
combined_eval = pd.concat([cgwb_eval, daily_eval], ignore_index=True)

well_mask_eval = np.zeros((105, 217), dtype=bool)
for cid in wel_cell_ids:
    node = cid - 1
    well_mask_eval[(node % (105 * 217)) // 217, (node % (105 * 217)) % 217] = True
river_mask_eval = (c_str_grid > 0)
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
print("EVALUATION METRICS FOR CYCLE 9 CHECKPOINT (50,000 COLLOCATION, BATCH 1024, PARTIAL REG RELAXATION)", flush=True)
print("="*60, flush=True)
for name, metrics in [('CGWB', cgwb_metrics), ('Daily', daily_metrics), ('Combined', combined_metrics), ('Near Wells/Rivers', near_metrics)]:
    print(f"\n{name} Dataset ({len(cgwb_eval) if name=='CGWB' else len(daily_eval) if name=='Daily' else len(combined_eval) if name=='Combined' else len(near_eval)} observations):", flush=True)
    print(f"  MODFLOW: R2={metrics['MODFLOW R2']:.4f}, RMSE={metrics['MODFLOW RMSE']:.4f} m, MAE={metrics['MODFLOW MAE']:.4f} m", flush=True)
    print(f"  PINN C9: R2={metrics['PINN R2']:.4f}, RMSE={metrics['PINN RMSE']:.4f} m, MAE={metrics['PINN MAE']:.4f} m", flush=True)
print("="*60 + "\n", flush=True)

# Save JSON file
import json
metrics_out = {
    'cgwb': cgwb_metrics,
    'daily': daily_metrics,
    'combined': combined_metrics,
    'near': near_metrics
}
with open(os.path.join(workspace, 'cycle_9_eval_metrics.json'), 'w') as f:
    json.dump(metrics_out, f, indent=4)

# ==========================================
# 4. Generate Mid-Timestep Heatmap
# ==========================================
print("Generating mid-timestep head heatmap...", flush=True)
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
inp_plot = [
    x_s.reshape(-1, 1), y_s.reshape(-1, 1), t_s.reshape(-1, 1)
]
for _ in range(75):
    inp_plot.append(zeros_plot.reshape(-1, 1))

pred_scaled = h_net[0].eval(m, inp_plot).reshape((105, 217))
pred_phys_raw = pred_scaled * scaler['hdiff'] + scaler['hl']

sp_mid = get_sp(t_mid)
alpha_mid = alpha_chd_grid[0]
h_chd_mid = h_chd_grid_3d[sp_mid, 0, :, :]

# Correctly apply hard CHD boundary condition ansatz for heatmap in physical units
h_pinn_mid = (1.0 - alpha_mid) * h_chd_mid + alpha_mid * pred_phys_raw

h_mf_mid[ibound_top <= 0] = np.nan
h_pinn_mid[ibound_top <= 0] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
vmin_mf = np.nanpercentile(h_mf_mid, 1)
vmax_mf = np.nanpercentile(h_mf_mid, 99)
vmin_pinn = np.nanpercentile(h_pinn_mid, 5)
vmax_pinn = np.nanpercentile(h_pinn_mid, 95)

im1 = axes[0].imshow(h_mf_mid, cmap='viridis', vmin=vmin_mf, vmax=vmax_mf)
axes[0].set_title(f'MODFLOW Head Variation at Mid-Timestep (t={t_mid:.1f} d)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Col Index')
axes[0].set_ylabel('Row Index')
fig.colorbar(im1, ax=axes[0], label='Head (m)')

im2 = axes[1].imshow(h_pinn_mid, cmap='viridis', vmin=vmin_pinn, vmax=vmax_pinn)
axes[1].set_title(f'PINN Cycle 3 Head Prediction (t={t_mid:.1f} d)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Col Index')
axes[1].set_ylabel('Row Index')
fig.colorbar(im2, ax=axes[1], label='Head (m)')

plt.tight_layout()
plt.savefig(os.path.join(workspace, 'mid_timestep_comparison_cycle_9.png'), dpi=300)
# Copy to artifacts directory
artifact_plot_path = r"C:\Users\Shreyansh\.gemini\antigravity\brain\4c84105f-6891-40f1-92e8-beb11d6d579c\mid_timestep_comparison_cycle_9.png"
plt.savefig(artifact_plot_path, dpi=300)
plt.close()
print("Saved mid-timestep comparison heatmap successfully!", flush=True)
