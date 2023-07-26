import pandas as pd

# Read benchmark comp times
df = pd.read_csv("assets/data/svr-info.csv")

# Drop columns
df.drop(["method", "units", "k_folds", "n_test", "n_train"], axis=1, inplace=True)

# Concat ranges and calculate ratios
df["r2"] = df.apply(lambda row: f"{row['r2_valid_mean']}±{row['r2_valid_sd'] * 2}", axis=1)
df["rmse"] = df.apply(lambda row: f"{row['rmse_valid_mean']}±{row['rmse_valid_sd'] * 2}", axis=1)

# Drop columns
df.drop(
    [
        "r2_test_mean",
        "r2_test_sd",
        "r2_valid_mean",
        "r2_valid_sd",
        "rmse_test_mean",
        "rmse_test_sd",
        "rmse_valid_mean",
        "rmse_valid_sd"
    ],
    axis=1,
    inplace=True
)

# Sort columns
df = df.sort_values(by=["program", "parameter", "kernel"])

# Change column headers
df.columns = [
    "Program",
    "Parameter",
    "Kernel",
    "Scaler",
    "R$^2$",
    "RMSE"
]

# Convert DataFrame to Markdown table
markdown_table = df.to_markdown(index=False)

# Write the Markdown table to a file
with open("draft/assets/pandoc/benchmark-times.md", "w") as file:
    file.write(markdown_table)

# Read benchmark comp times
df = pd.read_csv("assets/data/benchmark-times.csv")

# Concat ranges and calculate ratios
df["prange"] = df.apply(lambda row: f"{row['pmin']}-{row['pmax']}", axis=1)
df["trange"] = df.apply(lambda row: f"{row['tmin']}-{row['tmax']}", axis=1)
df["PPX/MGM"] = round(df["ppx"]/df["mgm"], 1)

# Drop columns
df.drop(["pmin", "pmax", "tmin", "tmax"], axis=1, inplace=True)

# Rearange df
columns_to_move = ["prange", "trange"]
df = df[["sample"] + columns_to_move + [col for col in df.columns if col not in ["sample"] + columns_to_move]]

# Change column headers
df.columns = [
    "Name",
    "P (GPa)",
    "T (K)",
    "Grid Size",
    "MAGEMin (s)",
    "Perple_X (s)",
    "PPX/MGM"
]

# Convert DataFrame to Markdown table
markdown_table = df.to_markdown(index=False, floatfmt=".1f")

# Write the Markdown table to a file
with open("draft/assets/pandoc/benchmark-times.md", "w") as file:
    file.write(markdown_table)

# Read benchmark compositions
df = pd.read_csv("assets/data/benchmark-comps.csv")

# Reorder the columns
new_order = [
    "NAME",
    "SIO2",
    "AL2O3",
    "CAO",
    "MGO",
    "FEO",
    "K2O",
    "NA2O",
    "TIO2",
    "FE2O3",
    "CR2O3",
    "H2O",
    "MGNUM",
    "REF"
]

df = df.reindex(columns=new_order)

# Drop Mg Number
df = df.drop(columns=["MGNUM", "REF"])

# Change column headers
col_headers = {
    "SIO2": "SiO$_2$",
    "AL2O3": "Al$_2$O$_3$",
    "CAO": "CaO",
    "MGO": "MgO",
    "FEO": "FeO",
    "K2O": "K$_2$O",
    "NA2O": "Na$_2$O",
    "TIO2": "TiO$_2$",
    "FE2O3": "O$_2$",
    "CR2O3": "Cr$_2$O$_3$",
    "H2O": "H$_2$O",
    "NAME": "Name",
}

df.rename(columns=col_headers, inplace=True)

# Convert DataFrame to Markdown table
markdown_table = df.to_markdown(index=False)

# Write the Markdown table to a file
with open("draft/assets/pandoc/benchmark-comps.md", "w") as file:
    file.write(markdown_table)