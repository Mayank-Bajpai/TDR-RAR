import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_dynamic_budget_plot(file_path, out_png):
    xls = pd.ExcelFile(file_path)
    df_mf = pd.read_excel(xls, 'MODFLOW Budget')
    df_pinn = pd.read_excel(xls, 'PINN Budget')

    # Filter out text rows
    df_mf = df_mf[pd.to_numeric(df_mf['Stress_Period'], errors='coerce').notnull()].copy()
    df_pinn = df_pinn[pd.to_numeric(df_pinn['Stress_Period'], errors='coerce').notnull()].copy()
    df_mf['Stress_Period'] = df_mf['Stress_Period'].astype(int)
    df_pinn['Stress_Period'] = df_pinn['Stress_Period'].astype(int)

    def calc_net(df):
        df['Net Storage'] = df['Storage_In'] + df['Storage_Out']
        df['Net Boundary Flow (CHD)'] = df['CHD_In'] + df['CHD_Out']
        df['Net Stream Leakage'] = df['Leakage_In'] + df['Leakage_Out']
        return df

    df_mf = calc_net(df_mf)
    df_pinn = calc_net(df_pinn)

    comps = ['Net Storage', 'Net Boundary Flow (CHD)', 'Net Stream Leakage']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    plt.style.use('ggplot')

    for j, comp in enumerate(comps):
        ax1 = axes[0, j]
        ax1.set_facecolor('whitesmoke')
        ax1.plot(df_mf['Stress_Period'], df_mf[comp], label='MODFLOW', color='royalblue', linewidth=2.5, marker='o')
        ax1.plot(df_pinn['Stress_Period'], df_pinn[comp], label='PINN', color='darkorange', linewidth=2.5, linestyle='--', marker='s')
        ax1.set_title(comp, fontsize=14, fontweight='bold', pad=10)
        if j == 0:
            ax1.set_ylabel('Net Volume (m³)', fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        ax2 = axes[1, j]
        ax2.set_facecolor('whitesmoke')
        diff = df_pinn[comp] - df_mf[comp]
        colors = ['#2ca02c' if val > 0 else '#d62728' for val in diff]
        ax2.axhline(0, color='black', linewidth=1.2, linestyle='-')
        ax2.bar(df_mf['Stress_Period'], diff, color=colors, width=0.6, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax2.set_title(f'Difference (PINN - MODFLOW)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Stress Period', fontsize=12, fontweight='bold')
        if j == 0:
            ax2.set_ylabel('Error Volume (m³)', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y') 
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2.set_xticks(range(0, 21, 2))

    plt.suptitle('Dynamic Water Budget Components: MODFLOW vs. PINN Surrogate', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # Usage example
    pass
