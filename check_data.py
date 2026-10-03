import pandas as pd

df = pd.read_csv('data/training_data.csv')
print("Dataset Shape:", df.shape)
print("PerformanceRating value counts:")
print(df['PerformanceRating'].value_counts())
print("\nMissing values:")
print(df.isnull().sum().sum())
print("\nColumns:")
print(list(df.columns))
