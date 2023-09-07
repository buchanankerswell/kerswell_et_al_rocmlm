import pandas as pd

# ML model parameters and performance metrics
df = pd.read_csv("assets/data/benchmark-mlms-metrics.csv")
df["inference_time_std"] = df["inference_time_std"] * 2
numeric_columns = df.select_dtypes(include=[float, int]).columns
df = df.groupby("model")[numeric_columns].mean().reset_index()
df.drop(
    ["k_folds",  "r2_mean", "r2_std", "training_time_mean", "training_time_std"],
    axis=1,
    inplace=True
)
df = df.sort_values(by=["inference_time_mean"])
df.columns = ["Model", "Efficiency (ms)", "$2\sigma$", "RMSE (g/cm$^3$)", "$2\sigma$"]
markdown_table = df.to_markdown(index=False, floatfmt=".3f")

with open("draft/assets/pandoc/benchmark-mlms-metrics.md", "w") as file:
    file.write(markdown_table)

# Benchmark comp times
df = pd.read_csv("assets/data/benchmark-gfem-efficiency.csv")
df.drop(["pmin", "pmax", "tmin", "tmax"], axis=1, inplace=True)
df.columns = ["Name", "Grid Size", "MAGEMin", "Perple_X"]
markdown_table = df.to_markdown(index=False, floatfmt=".1f")

with open("draft/assets/pandoc/benchmark-gfem-efficiency.md", "w") as file:
    file.write(markdown_table)

# Benchmark compositions
df = pd.read_csv("assets/data/benchmark-samples.csv")
new_order = [
    "NAME", "SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O",
    "TIO2", "FE2O3", "CR2O3", "H2O", "MGNUM", "REF"
]
df = df.reindex(columns=new_order)
df = df.drop(columns=["MGNUM", "REF"])
col_headers = {
    "SIO2": "SiO$_2$", "AL2O3": "Al$_2$O$_3$", "CAO": "CaO", "MGO": "MgO", "FEO": "FeO",
    "K2O": "K$_2$O", "NA2O": "Na$_2$O", "TIO2": "TiO$_2$", "FE2O3": "O$_2$",
    "CR2O3": "Cr$_2$O$_3$", "H2O": "H$_2$O", "NAME": "Name",
}
df.rename(columns=col_headers, inplace=True)
markdown_table = df.to_markdown(index=False)

with open("draft/assets/pandoc/benchmark-samples.md", "w") as file:
    file.write(markdown_table)