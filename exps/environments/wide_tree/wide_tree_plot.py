"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.environments.utilities import get_color, get_linestyle

df = pd.read_pickle(f"wide_tree_results.pkl")

set_figure_params(serif=True)
fig, axes = plt.subplots(ncols=1, nrows=1)
plot_df_key(
    df=df,
    axes=axes,
    key="value_evaluation",
    get_color=get_color,
    get_linestyle=get_linestyle,
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"Wide-Tree")

handles, labels = axes.get_legend_handles_labels()
idx = labels.index(r"\textbf{Q-REPS}")
handles = [handles[idx]] + handles[:idx] + handles[idx + 1 :]
labels = [labels[idx]] + labels[:idx] + labels[idx + 1 :]

# plt.legend(loc="best", frameon=False)
plt.legend(handles, labels, bbox_to_anchor=(0.7, 0.32), loc="lower left", frameon=False)
plt.savefig("wide_tree_results.pdf", bbox_inches="tight")
