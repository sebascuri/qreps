"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.effect_of_alpha.utilities import get_color, get_linestyle
import seaborn as sns

palette = sns.color_palette(n_colors=10)


df = pd.read_pickle("alpha_results.pkl")
df = df[df.name != "ExactREPS-1"]
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
plt.title(r"Effect of $\alpha$")

# plt.plot(0, 0, linestyle="solid", color="k", label="QREPS")
# plt.plot(0, 0, linestyle="dashed", color="k", label="REPS")
plt.plot(0, 0, linestyle="solid", color=palette[4], label=r"$\alpha=10$")
plt.plot(0, 0, linestyle="solid", color=palette[3], label=r"$\alpha=3$")
plt.plot(0, 0, linestyle="solid", color=palette[0], label=r"$\alpha=1$")
plt.plot(0, 0, linestyle="solid", color=palette[2], label=r"$\alpha=.3$")
plt.plot(0, 0, linestyle="solid", color=palette[1], label=r"$\alpha=.1$")

handles, labels = axes.get_legend_handles_labels()
plt.figlegend(
    handles[-5:],
    labels[-5:],
    bbox_to_anchor=(0.99, 0.18),
    loc="lower right",
    frameon=False,
    # shadow=False,
    ncol=1,
    fancybox=False,
)
plt.tight_layout(pad=0.2)

# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
plt.savefig("alpha_results.pdf", bbox_inches="tight")
