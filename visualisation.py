import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tsinfer_path = os.path.abspath("/well/kelleher/users/uuc395/tsinfer")
sys.path.append(tsinfer_path)
import tsinfer


def visualize_perf(
    folder, prefix, versions=["0.4", "1.0"], colors=["darkorange", "royalblue"]
):
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
        perf_df["version"] = version
        df_list.append(perf_df)

        linesweep_df = pd.read_csv(
            os.path.join(folder, f"{prefix}-{version}-linesweep.csv")
        )
        ancestors_per_epoch = linesweep_df.ancestors_per_epoch.values
        ax_linesweep.scatter(
            range(len(ancestors_per_epoch)),
            ancestors_per_epoch,
            color=colors[i],
            label=f"{version}",
            s=10,
            alpha=0.7,
        )

    perf_df = pd.concat(df_list)
    ax_linesweep.set_yscale("log")
    ax_linesweep.set_ylabel("Ancestors per epoch")
    ax_linesweep.set_xlabel("Epoch")
    ax_linesweep.set_title("Linesweep results: number of ancestors per epoch")
    ax_linesweep.legend(title="Version")

    steps = perf_df["step"].unique()
    bottom_gs = gs[1].subgridspec(1, 3, wspace=0.4)

    for i, step in enumerate(steps):
        step_data = perf_df[perf_df["step"] == step]
        ax = fig.add_subplot(bottom_gs[i])
        x = np.arange(2)
        width = 0.8 / len(versions)

        for j, version in enumerate(versions):
            version_data = step_data[step_data["version"] == version]
            if not version_data.empty:
                pos = x - 0.4 + width * (j + 0.5)
                values = [
                    version_data["wall_time"].values[0],
                    version_data["cpu_time"].values[0],
                ]
                bars = ax.bar(pos, values, width, label=f"{version}", color=colors[j])

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        rotation=0,
                        fontsize=9,
                    )

        ax.set_title(f"{step}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Wall time (s)", "CPU time (s)"])
        ax.set_ylabel("Time (s)")

        if i == 0:
            ax.legend(title="Version")

    plt.tight_layout()
    plt.show()


def plot_ancestor_boxplot(
    df,
    cutoffs=None,
    var="span",
    type="site",
    title="Ancestor lengths",
    y_log=False,
    save_path=None,
    color_dict={
        "new": "#d62728",  # red
        "old": "#ff7f0e",  # orange
        "true": "#1f77b4",  # blue
    },
):
    df = df.copy()  # To avoid SettingWithCopyWarning
    df = df.drop_duplicates(subset="inferred_index", keep="first")
    if cutoffs is None:
        cutoffs = np.unique(np.percentile(df["inferred_time"], np.linspace(0, 100, 9)))

    if type == "site":
        y_units = "sites"
    elif type == "pos":
        y_units = "bp"
    else:
        raise ValueError("type must be 'site' or 'pos'")
    if var == "span":
        vars = [
            f"true_{type}_span",
            f"inferred_new_{type}_span",
            f"inferred_old_{type}_span",
        ]
        var_labels = ["True", "Inferred (v1.0)", "Inferred (v0.4)"]
        colors = [color_dict["true"], color_dict["new"], color_dict["old"]]
    elif var == "overshoot":
        vars = [f"{type}_max_overshoot", f"old_{type}_max_overshoot"]
        var_labels = ["1.0a", "0.4"]
        colors = ["royalblue", "darkorange"]
    else:
        raise ValueError("var must be 'span' or 'overlap_ratio'")

    df["frequency_bin"] = pd.cut(df["inferred_time"], bins=cutoffs, include_lowest=True)
    df["frequency_bin"] = df["frequency_bin"].apply(
        lambda x: f"({x.left:.2f}, {x.right:.2f}]"
    )
    # Create a mapping from variable names to their formatted labels
    var_label_mapping = dict(zip(vars, var_labels))

    # Melt the dataframe and map the 'type' column to the formatted labels
    lengths_df = pd.melt(
        df,
        id_vars=["frequency_bin"],
        value_vars=vars,
        var_name="type",
        value_name="value",
    )
    lengths_df["type"] = lengths_df["type"].map(var_label_mapping)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 7), gridspec_kw={"height_ratios": [4, 1]}
    )

    # Use colors from var_dict in the boxplot palette, mapped by formatted labels
    palette = {label: color for label, color in zip(var_labels, colors)}
    sns.boxplot(
        x="frequency_bin",
        y="value",
        hue="type",
        data=lengths_df,
        palette=palette,
        saturation=1,
        hue_order=var_labels,
        ax=ax1,
    )

    # Adjust legend position to the right of the plot
    ax1.legend(title="Version", loc="upper left", bbox_to_anchor=(1.02, 1))

    ax1.set_xlabel("Ancestor age interval (frequency)")
    ax1.set_ylabel(f"Ancestor length overshoot ({y_units})")
    ax1.set_title(title)
    if y_log:
        ax1.set_yscale("log")

    # Plot quantile counts
    quantile_counts = df["frequency_bin"].value_counts(sort=False)
    sns.barplot(
        x=quantile_counts.index,
        y=quantile_counts.values,
        ax=ax2,
        color="#ced4da",
        linewidth=1,
        edgecolor="black",
    )
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax2.set_xlabel("")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Add margin for legend on the right

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free up memory
    else:
        plt.show()

