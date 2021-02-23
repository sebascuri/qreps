"""Python Script Template."""
import pandas as pd

from exps.utilities import parse_arguments

from exps.plotting import plot_df_key, set_figure_params
from exps.effect_of_algorithm.utilities import (
    get_color,
    get_linestyle,
)

import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette(n_colors=10)

args = parse_arguments()
df = pd.read_pickle("algorithm_results.pkl")
df = df[df.name != "Mini-Batch-REPS"]
df = df[df.name != "Biased-REPS"]
df = df[df.name != "SaddleQREPS"]

set_figure_params(serif=True, fontsize=12)
fig, axes = plt.subplots(ncols=1, nrows=1)
fig.set_size_inches(6.75 / 2, 2.0)

plot_df_key(
    df=df,
    axes=axes,
    key="value_evaluation",
    get_color=get_color,
    get_linestyle=get_linestyle,
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title(r"Min-Max Optimization")

plt.plot(0, 0, linestyle="solid", color="k", label="Sample-based")
plt.plot(0, 0, linestyle="dashed", color="k", label="Model-based")

plt.plot(0, 0, linestyle="solid", color=palette[0], label=r"QREPS")
plt.plot(0, 0, linestyle="solid", color=palette[1], label=r"REPS")

handles, labels = axes.get_legend_handles_labels()
plt.figlegend(
    handles[-4:],
    labels[-4:],
    bbox_to_anchor=(1.0, 0.21),
    loc="lower left",
    # frameon=True,
    shadow=True,
    ncol=1,
    fancybox=True,
)
plt.tight_layout(pad=0.2)


plt.savefig(f"algorithm_results.pdf", bbox_inches="tight")
