from mlcroissant import Dataset

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
