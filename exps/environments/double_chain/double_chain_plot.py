"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.environments.utilities import get_color, get_linestyle


df = pd.read_pickle("double_chain_results.pkl")
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
plt.title(f"Double Chain")
# plt.legend(loc="best", frameon=False)
plt.legend(bbox_to_anchor=(0.52, 0.3), loc="lower left", frameon=False)
plt.savefig("double_chain_results.pdf", bbox_inches="tight")
