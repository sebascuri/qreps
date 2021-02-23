"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.bias.utilities import get_color, get_linestyle
import seaborn as sns

palette = sns.color_palette(n_colors=10)


df = pd.read_pickle(f"bias_results.pkl")

set_figure_params(serif=True, fontsize=12)
fig, axes = plt.subplots(ncols=1, nrows=1)
fig.set_size_inches(6.75 / 2, 2.5)
plot_df_key(
    df=df,
    axes=axes,
    key="value_evaluation",
    get_color=get_color,
    get_linestyle=get_linestyle,
)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title(r"Effect of $\eta$")

plt.plot(0, 0, linestyle="solid", color="k", label="QREPS")
plt.plot(0, 0, linestyle="dashed", color="k", label=r"QREPS${}^*$")
plt.plot(0, 0, linestyle="solid", color=palette[6], label=r"$\eta=10$")
# plt.plot(0, 0, linestyle="solid", color=palette[3], label=r"$\eta=3$")
plt.plot(0, 0, linestyle="solid", color=palette[0], label=r"$\eta=1$")
# plt.plot(0, 0, linestyle="solid", color=palette[2], label=r"$\eta=.3$")
plt.plot(0, 0, linestyle="solid", color=palette[7], label=r"$\eta=.1$")

handles, labels = axes.get_legend_handles_labels()
plt.legend(
    handles[-5:],
    labels[-5:],
    # bbox_to_anchor=(1.0, 0.22),
    loc="best",
    frameon=False,
    shadow=False,
    ncol=1,
    fancybox=False,
)
# plt.xticks([0, 5, 10, 15, 20])
plt.tight_layout(pad=0.2)

# # plt.legend(loc="lower right", frameon=False, ncol=2)
# plt.legend(
#     bbox_to_anchor=(0.5, 0.4),
#     loc="lower left",
#     frameon=False,
#     ncol=2,
#     columnspacing=0.2,
# )
plt.savefig(f"bias_results.pdf", bbox_inches="tight")
