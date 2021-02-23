"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.environments.utilities import get_color, get_linestyle


ENVIRONMENTS = {
    "cart_pole": 200,
    "double_chain": 10,
    "river_swim": 0.8,
    "single_chain": 10,
    "two_state_deterministic": 2,
    "two_state_stochastic": 1,
    "wide_tree": 0.38,
    "windy_grid_world": 50,
}

set_figure_params(serif=True, fontsize=12)
fig, axes = plt.subplots(ncols=4, nrows=2, sharey=True)
fig.set_size_inches(6.75, 2)

for i, (environment, max_value) in enumerate(ENVIRONMENTS.items()):
    row, col = i // 4, i % 4

    df = pd.read_pickle(f"{environment}/{environment}_results.pkl")
    df = df[df.name != "DQN-delayed"]
    df = df[df.name != "REINFORCE"]
    plot_df_key(
        df=df,
        axes=axes[row, col],
        key="value_evaluation" if environment != "cart_pole" else "train_return",
        get_color=get_color,
        get_linestyle=get_linestyle,
        max_value=max_value,
    )
    if environment == "two_state_deterministic":
        title = "Two State D"
    elif environment == "two_state_stochastic":
        title = "Two State S"
    elif environment == "windy_grid_world":
        title = "Grid World"
    else:
        title = " ".join(environment.split("_")).title()

    axes[row, col].set_title(title, fontsize=11)

for col in range(4):
    for row in range(2):
        plt.setp(axes[row, col].get_xticklabels(), visible=True, fontsize=9)
for row in range(2):
    plt.setp(axes[row, 0].get_yticklabels(), visible=False)

fig.text(0.5, -0.03, "Episode", ha="center", fontsize=14)
fig.text(-0.03, 0.5, "Normalized Return", va="center", rotation="vertical", fontsize=14)

handles, labels = axes[0, 0].get_legend_handles_labels()
idx = labels.index(r"\textbf{Q-REPS}")
handles = [handles[idx]] + handles[:idx] + handles[idx + 1 :]
labels = [labels[idx]] + labels[:idx] + labels[idx + 1 :]
# plt.figlegend(
#     handles,
#     labels,
#     bbox_to_anchor=(0.08, 1.0),
#     loc="lower left",
#     # frameon=True,
#     shadow=True,
#     ncol=5,
#     fancybox=True,
# )

plt.figlegend(
    handles,
    labels,
    bbox_to_anchor=(1.0, 0.24),
    loc="lower left",
    # frameon=True,
    shadow=True,
    ncol=1,
    fancybox=True,
)
plt.tight_layout(pad=0.2)
plt.savefig("environment_results.pdf", bbox_inches="tight")
plt.show()
