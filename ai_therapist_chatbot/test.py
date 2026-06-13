from mlcroissant import Dataset
import pandas as pd

file_path = "mental-health-metadata.json"
ds = Dataset(jsonld=file_path)

# Read records or metadata
records = ds.records("mental_health_conversations.csv")

print("Printing first 5 records")
for i, record in enumerate(records):
    print(f"\n Records: {i+1}")
    print(record)

    if i>=4:
        break
df = pd.DataFrame(records)

df.drop(columns=["mental_health_conversations.csv/statement"])

df.to_csv('mental_health_convo.csv')

print(df.info())
print(df.head())
# print(df.sample(5))
# print(f"Null values information, {df.isna().sum()}")

# df.dropna()

print(f"updated info: \n", df.info())