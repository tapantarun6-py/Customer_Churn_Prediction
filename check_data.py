import pandas as pd

df = pd.read_csv("data/churn.csv")
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
