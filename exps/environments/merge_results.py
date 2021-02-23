"""Python Script Template."""
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser("Merge results files.")
parser.add_argument("env", type=str)
args = parser.parse_args()

df = pd.DataFrame()
for file in os.listdir("."):
    if file == f"{args.env}_results.pkl":
        continue
    if args.env in file and file.endswith(".pkl"):
        df = pd.concat((df, pd.read_pickle(file)), sort=False)

print(df)
df.to_pickle(f"{args.env}/{args.env}_results.pkl", protocol=4)
