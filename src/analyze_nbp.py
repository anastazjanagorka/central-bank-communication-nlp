import pandas as pd

df = pd.read_csv("data/nbp_communications.csv")

print(df.shape)
print(df.columns)
print(df.head())

print("\nReport statistics:")

df["word_count"] = df["clean_text"].str.split().str.len()
print(df["word_count"].describe())

# Calculate average communication length by year
yearly_length = df.groupby("year")["word_count"].mean()

print("\nAverage communication length by year:")
print(yearly_length)