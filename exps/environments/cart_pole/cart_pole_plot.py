"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.environments.utilities import get_color, get_linestyle


df = pd.read_pickle("cart_pole_results.pkl")
df = df[df.time < 25]
# set_figure_params(serif=True)
fig, axes = plt.subplots(ncols=1, nrows=1)
plot_df_key(
    df=df,
    axes=axes,
    key="train_return",
    get_color=get_color,
    get_linestyle=get_linestyle,
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title(f"Cart-Pole")
plt.legend(loc="best", frameon=False)
# plt.legend(bbox_to_anchor=(0.7, 0.32), loc="lower left", frameon=False)
plt.savefig("cart_pole_results.pdf", bbox_inches="tight")
