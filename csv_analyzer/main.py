import pandas as pd

from csv_analyzer.columns_analyzer import profile_dataframe
from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client

df = pd.read_csv("data/insurance 2.csv")
profiles = profile_dataframe(df)
print(profiles[0])

embeddings_client = get_multilingual_embeddings_client()

print()