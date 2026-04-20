import pandas as pd

df = pd.read_excel("data/Dataset_Challenge_1.xlsx")

print(df.head())

inputs = df["English Source"].tolist()
refs = df["Reference Translation"].tolist()

print(inputs[:2])
print(refs[:2])

print("Total rows:", len(df))