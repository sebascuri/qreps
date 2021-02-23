"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.action_gap.utilities import get_color, get_linestyle
import seaborn as sns
import numpy as np

set_figure_params(serif=True, fontsize=12)
palette = sns.color_palette(n_colors=10)

df = pd.read_pickle("action_gap_results.pkl")
df["a_0"] = df.q_00 - df.q_01

action_gap_exact, action_gap_saddle = {}, {}

for name, value in df.groupby("name"):
    alpha = float(name.split("-")[1])
    if "Exact" in name:
        action_gap_exact[alpha] = value.a_0.max()
    else:
        action_gap_saddle[alpha] = value.a_0.max()

alpha = np.array(sorted(action_gap_saddle.keys()))
action_gap_exact = [action_gap_exact[a] for a in alpha]
action_gap_saddle = [action_gap_saddle[a] for a in alpha]

fig, axes = plt.subplots(ncols=2, nrows=1)
fig.set_size_inches(6.75, 2.5)

axes[0].plot(alpha, action_gap_saddle, color="k", linestyle="solid", label="QREPS")
axes[0].plot(
    alpha,
    action_gap_exact,
    color=palette[-2],
    linestyle="dashed",
    label=r"QREPS${}^*$",
)
axes[0].plot(
    alpha, 1 / alpha, color=palette[-1], linestyle="dotted", label=r"$1 / \alpha$"
)
axes[0].set_yscale("log")
axes[0].set_xscale("log")

axes[0].set_title(r"Effect of $\alpha$ on action gap")
axes[0].set_xlabel(r"$\alpha$")
axes[0].set_ylabel(r"$Q(x_0, \mathrm{stay}) - Q(x_0, \mathrm{go})$")
axes[0].legend(
    loc="lower left",
    bbox_to_anchor=(0, 0),
    frameon=False,
    shadow=False,
    ncol=1,
    fancybox=False,
)

# plt.show()

plot_df_key(
    df=df,
    axes=axes[1],
    key="value_evaluation",
    get_color=get_color,
    get_linestyle=get_linestyle,
)

axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Average Reward")
axes[1].set_title(r"Effect of $\alpha$ on learning")

# axes[1].plot(0, 0, linestyle="solid", color="k", label="QREPS")
# axes[1].plot(0, 0, linestyle="dashed", color="k", label="QREPS*")
axes[1].plot(0, 0, linestyle="solid", color=palette[1], label=r"$\alpha=1000$")
axes[1].plot(0, 0, linestyle="solid", color=palette[2], label=r"$\alpha=100$")
axes[1].plot(0, 0, linestyle="solid", color=palette[3], label=r"$\alpha=10$")
axes[1].plot(0, 0, linestyle="solid", color=palette[0], label=r"$\alpha=1$")
axes[1].plot(0, 0, linestyle="solid", color=palette[4], label=r"$\alpha=0.1$")

handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(
    handles[-5:],
    labels[-5:],
    # bbox_to_anchor=(1.2, 0.18),
    loc="lower right",
    frameon=False,
    shadow=False,
    ncol=1,
    fancybox=False,
    handlelength=1.0,
    labelspacing=0.1,
)
plt.tight_layout(pad=0.2)
plt.savefig("action_gap_results.pdf", bbox_inches="tight")
