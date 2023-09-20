import pandas as pd

# RocML performance metrics
df = pd.read_csv("assets/data/benchmark-rocmls-performance.csv")

# Use 2 sigma standard deviation
df["inference_time_std"] = df["inference_time_std"] * 2

# Select numeric columns
numeric_columns = df.select_dtypes(include=[float, int]).columns

# Group by model and compute mean values
df = df.groupby("model")[numeric_columns].mean().reset_index()

# Drop metrics
df.drop(
    ["k_folds",  "r2_mean", "r2_std", "training_time_mean", "training_time_std"],
    axis=1,
    inplace=True
)

# Sort by prediction efficiency
df = df.sort_values(by=["inference_time_mean"])

# Rename columns
df.columns = ["Model", "Efficiency (ms)", "$2\sigma$", "RMSE (g/cm$^3$)", "$2\sigma$"]

# Generate markdown table
markdown_table = df.to_markdown(index=False, floatfmt=".3f")

# Write markdown table
with open("draft/assets/pandoc/benchmark-rocmls-performance.md", "w") as file:
    file.write(markdown_table)

# Benchmark effeciency
df = pd.read_csv("assets/data/benchmark-gfem-efficiency.csv")

# Rename columns
df.columns = ["Sample", "Program", "Dataset Size", "Elapes Time (s)"]

# Generate markdown table
markdown_table = df.to_markdown(index=False, floatfmt=".1f")

# Write markdown table
with open("draft/assets/pandoc/benchmark-gfem-efficiency.md", "w") as file:
    file.write(markdown_table)

# Benchmark compositions
df = pd.read_csv("assets/data/benchmark-samples.csv")

# Drop references
df = df.drop(columns=["REF"])

# Rename columns
col_headers = {
    "NAME": "Name", "SIO2": "SiO$_2$", "AL2O3": "Al$_2$O$_3$", "CAO": "CaO", "MGO": "MgO",
    "FEO": "FeO", "K2O": "K$_2$O", "NA2O": "Na$_2$O", "TIO2": "TiO$_2$", "FE2O3": "O$_2$",
    "CR2O3": "Cr$_2$O$_3$", "H2O": "H$_2$O"
}

df.rename(columns=col_headers, inplace=True)

# Generate markdown table
markdown_table = df.to_markdown(index=False)

# Write markdown table
with open("draft/assets/pandoc/benchmark-samples.md", "w") as file:
    file.write(markdown_table)

print("write-markdown-tables.py done!")