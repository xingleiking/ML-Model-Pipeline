import pandas as pd

# Read input CSV file
df = pd.read_csv("7_3.csv")

# Extract the label column
labels = df["label"]

# Compute minimum and maximum values
min_val = labels.min()
max_val = labels.max()

# Apply Min-Max normalization
df["normalized_label"] = (labels - min_val) / (max_val - min_val)

# Save to a new CSV file (keep original label and add normalized column)
df.to_csv("normalized.csv", index=False)

print("Normalization completed. Results saved to normalized.csv")
