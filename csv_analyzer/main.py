import pandas as pd

from csv_analyzer.columns_analyzer import profile_dataframe

df = pd.read_csv("data/insurance 2.csv")
profiles = profile_dataframe(df)
print(profiles[0])