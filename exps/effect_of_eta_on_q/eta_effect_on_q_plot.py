"""Python Script Template."""
import pandas as pd
import matplotlib.pyplot as plt
from exps.plotting import plot_df_key, set_figure_params
from exps.effect_of_eta.utilities import get_color, get_linestyle
import seaborn as sns

palette = sns.color_palette(n_colors=10)


df = pd.read_pickle(f"eta_on_q_results.pkl")
df["eta"] = 0
for name, _ in df.groupby("name"):
    df.loc[df.name == name, "eta"] = float(name.split("-")[1])

df["q_max"] = df[["q_00", "q_01", "q_10", "q_11"]].max(axis=1) - df[
    ["q_00", "q_01", "q_10", "q_11"]
].min(axis=1)

df["qbar_max"] = df[["qbar_00", "qbar_01", "qbar_10", "qbar_11"]].abs().max(axis=1)

print(df.groupby(["eta"]).q_max.max())
print(df.groupby(["eta"]).qbar_max.max())
