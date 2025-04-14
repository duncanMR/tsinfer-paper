import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
sys.path.append(tsinfer_path)
import tsinfer

def visualize_perf(folder, prefix, versions=['0.4', '1.0'], colors=['darkorange', 'royalblue']):
    """
    Create a multi-panel plot visualizing tsinfer performance.
    
    Parameters:
    - folder: directory containing performance data files
    - prefix: prefix for the filenames
    - versions: list of tsinfer versions to compare
    - colors: list of colors for plotting each version
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    ax_linesweep = fig.add_subplot(gs[0])
    
    df_list = []
    
    for i, version in enumerate(versions):
        perf_df = pd.read_csv(os.path.join(folder, f"{prefix}-{version}-perf.csv"))
        perf_df['version'] = version
        df_list.append(perf_df)
        
        linesweep_df = pd.read_csv(os.path.join(folder, f"{prefix}-{version}-linesweep.csv"))
        ancestors_per_epoch = linesweep_df.ancestors_per_epoch.values
        ax_linesweep.scatter(range(len(ancestors_per_epoch)), ancestors_per_epoch,
                    color=colors[i], label=f'{version}', s=10, alpha=0.7)
    
    perf_df = pd.concat(df_list)
    ax_linesweep.set_yscale('log')
    ax_linesweep.set_ylabel('Ancestors per epoch')
    ax_linesweep.set_xlabel('Epoch')
    ax_linesweep.set_title('Linesweep results: number of ancestors per epoch')
    ax_linesweep.legend(title='Version')
    
    steps = perf_df['step'].unique()
    bottom_gs = gs[1].subgridspec(1, 3, wspace=0.4)
    
    for i, step in enumerate(steps):
        step_data = perf_df[perf_df['step'] == step]
        ax = fig.add_subplot(bottom_gs[i])
        x = np.arange(2)
        width = 0.8 / len(versions)
        
        for j, version in enumerate(versions):
            version_data = step_data[step_data['version'] == version]
            if not version_data.empty:
                pos = x - 0.4 + width * (j + 0.5)
                values = [version_data['wall_time'].values[0], version_data['cpu_time'].values[0]]
                bars = ax.bar(pos, values, width, label=f'{version}', color=colors[j])
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', rotation=0, fontsize=9)
        
        ax.set_title(f'{step}')
        ax.set_xticks(x)
        ax.set_xticklabels(['Wall time (s)', 'CPU time (s)'])
        ax.set_ylabel('Time (s)')
        
        if i == 0:
            ax.legend(title='Version')
    
    plt.tight_layout()
    plt.show()
