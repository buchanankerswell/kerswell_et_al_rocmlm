import os
import re
import shutil
import numpy as np
import pandas as pd

def main():
    """
    """
    data_dir = "assets/data"
    df_path = f"{data_dir}/benchmark-samples-pca.csv"
    df_synth_path = f"{data_dir}/synthetic-samples-benchmarks.csv"

    if (os.path.exists(df_path) and os.path.exists(df_synth_path)):
        print("Writing benchmark-samples.md")

        # Benchmark compositions
        df = pd.read_csv(df_path)
        df_synth = pd.read_csv(df_synth_path)

        # Drop columns
        df = df.drop(columns=["REFERENCE", "FE2O3", "PC1", "PC2", "R_TIO2", "F_MELT_BATCH",
                              "F_MELT_FRAC", "D_BATCH"])
        df_synth = df_synth.drop(columns=["FE2O3", "PC1", "PC2", "R_TIO2", "F_MELT_BATCH",
                                          "F_MELT_FRAC", "D_BATCH"])

        # Get synth samples
        df_synth = df_synth[df_synth["SAMPLEID"].isin(["sm000", "sm128"])]
        df_synth.loc[df_synth["SAMPLEID"] == "sm000", "SAMPLEID"] = "DSUM"
        df_synth.loc[df_synth["SAMPLEID"] == "sm128", "SAMPLEID"] = "PSUM"

        # Combine data
        df_combined = pd.concat([df, df_synth], ignore_index=True).sort_values(by="SAMPLEID")

        # Subset columns
        cols = ["SAMPLEID", "SIO2", "TIO2", "AL2O3", "FEO", "MGO", "CAO", "NA2O", "D_FRAC"]
        df_combined = df_combined[cols]

        # Rename columns
        col_headers = {"SAMPLEID": "Sample", "SIO2": "SiO$_2^{*}$", "TIO2": "TiO$_2^{*}$",
                       "AL2O3": "Al$_2$O$_3^{*}$", "FEO": "FeOT$^{*}$", "MGO": "MgO$^{*}$",
                       "CAO": "CaO$^{*}$", "NA2O": "Na$_2$O$^{*}$", "D_FRAC": "$\\xi$"}

        df_combined.rename(columns=col_headers, inplace=True)
        df_combined.sort_values(by="$\\xi$", inplace=True)

        # Generate markdown table
        markdown_table = df_combined.to_markdown(index=False, floatfmt=".3g")

        # Write markdown table
        with open("draft/benchmark-samples.md", "w") as file:
            file.write(f"{markdown_table}")

    else:
        print(f"Warning: {df_path} does not exist!")

    if os.path.exists(f"{data_dir}/earthchem-counts.csv"):
        print("Writing earthchem-counts.md")

        # Earthchem imputed data counts
        df = pd.read_csv(f"{data_dir}/earthchem-counts.csv")

        # Subset oxides
        oxides = {"SIO2": "SiO$_2$", "TIO2": "TiO$_2$", "AL2O3": "Al$_2$O$_3$", "FEO": "FeOT",
                  "MGO": "MgO", "CAO": "CaO", "NA2O": "Na$_2$O"}
        oxide_order = ["SiO$_2$", "TiO$_2$", "Al$_2$O$_3$", "FeOT", "MgO", "CaO", "Na$_2$O"]
        df["column"] = df["column"].map(oxides)
        df = df.dropna(subset=["column"])
        df = df.set_index("column").reindex(oxide_order).reset_index()

        # Rename columns
        df.loc[df["column"] == "FEO", "column"] = "FEOT"
        col_headers = {"column": "Oxide", "min": "Min$^{*}$", "max": "Max$^{*}$",
                       "mean": "Mean$^{*}$", "std": "Std$^{*}$", "median": "Median$^{*}$",
                       "iqr": "IQR$^{*}$", "measured": "Measured", "missing": "Missing",
                       "imputed": "Imputed"}
        df.rename(columns=col_headers, inplace=True)

        # Generate markdown table
        markdown_table = df.to_markdown(index=False, floatfmt=".3g")

        # Write markdown table
        with open("draft/earthchem-counts.md", "w") as file:
            file.write(f"{markdown_table}")

    else:
        print(f"Warning: {data_dir}/earthchem-counts.csv does not exist!")

    if os.path.exists(f"{data_dir}/rocmlm-config.csv"):
        print("Writing rocmlm-config.md")

        # RocMLM config
        df = pd.read_csv(f"{data_dir}/rocmlm-config.csv")
        df.fillna("", inplace=True)

        # Rename columns
        col_headers = {"model": "Model", "hyperparam": "Hyperparameter",
                       "hyperparams_value": "Value", "hyperparams_tuned": "Tuned"}
        df.rename(columns=col_headers, inplace=True)

        # Generate markdown table
        markdown_table = df.to_markdown(index=False)

        # Write markdown table
        with open("draft/rocmlm-config.md", "w") as file:
            file.write(f"{markdown_table}")

    else:
        print(f"Warning: {data_dir}/rocmlm-config.csv does not exist!")

    if os.path.exists(f"{data_dir}/rocmlm-performance.csv"):
        print("Writing rocmlm-performance.md")

        # RocMLM performance metrics
        df = pd.read_csv(f"{data_dir}/rocmlm-performance.csv")

        # Filter models
        df = df[df["sample"].isin(["SYNTH129", "SYNTH65", "SYNTH33", "benchmark"])]

        # Transform units
        df["training_time_mean"] = df["training_time_mean"] * 1000
        df["inference_time_mean"] = df["inference_time_mean"] * 1000
        df["size"] = np.sqrt(df["size"])

        # Round Mb size
        df["model_size_mb"] = round(df["model_size_mb"], 3)
        df["model_size_mb"] = df["model_size_mb"].apply(lambda x: f"{x:.3g}")

        # Get X resolution
        def get_x_res(row):
            if row["sample"].startswith("SYNTH") and row["sample"][5:].isdigit():
                return int(row["sample"][5:]) - 1
            else:
                return 2

        df["x_res"] = df.apply(get_x_res, axis=1)

        # Rename sample
        def rename_sample(row):
            if row["sample"].startswith("SYNTH") and row["sample"][5:].isdigit():
                return row["sample"][:5]
            else:
                return "Bench"

        df["sample"] = df.apply(rename_sample, axis=1)

        # Drop columns
        df.drop(["n_targets", "k_folds"], axis=1, inplace=True)
        df.drop([col for col in df.columns if "r2_" in col], axis=1, inplace=True)
        df.drop([col for col in df.columns if "_test" in col], axis=1, inplace=True)

        # Sort by prediction efficiency
        df.sort_values(by=["inference_time_mean"], inplace=True)

        # Multiply _std columns by 2 and combine with _mean columns
        for col in df.columns:
            if "_mean" in col:
                df[col] = df[col].apply(lambda x: f"{x:.2g}")
                std_col = col.replace("_mean", "_std")
                if std_col in df.columns:
                    df[std_col] = df[std_col].apply(lambda x: f"{2 * x:.2g}")

        # Drop the std_ columns
        df.drop([col for col in df.columns if "_std" in col], axis=1, inplace=True)

        # Drop program column
        df.drop(["program"], axis=1, inplace=True)

        # Order df
        df = df[["model", "size", "x_res", "training_time_mean", "inference_time_mean",
                 "rmse_val_mean_rho", "rmse_val_mean_Vp", "rmse_val_mean_Vs",
                 "model_size_mb"]]

        # Sort df
        df = df.sort_values(by=["model", "size", "x_res"])

        # Rename columns
        df.columns = ["Model", "PT Res (pts)", "X Res (pts)", "Train (ms)", "Predict (ms)",
                      "RMSE rho (g/cm$^3$)", "RMSE Vp (km/s)", "RMSE Vs (km/s)",
                      "Filesize (Mb)"]

        # Generate markdown table
        markdown_table = df.to_markdown(index=False)

        # Write markdown table
        with open("draft/rocmlm-performance.md", "w") as file:
            file.write(f"{markdown_table}")

    else:
        print(f"Warning: {data_dir}/rocmlm-performance.csv does not exist!")

    if os.path.exists(f"{data_dir}/gfem-efficiency.csv"):
        print("Writing gfem-efficiency.md")

        # Benchmark effeciency
        df = pd.read_csv(f"{data_dir}/gfem-efficiency.csv")
        df.drop(["dataset"], axis=1, inplace=True)

        # Rename columns
        df.columns = ["Sample", "Program", "Model Size", "Elapsed Time (s)"]

        # Generate markdown table
        markdown_table = df.to_markdown(index=False, floatfmt=".1f")

        # Write markdown table
        with open("draft/gfem-efficiency.md", "w") as file:
            file.write(f"{markdown_table}")

    else:
        print(f"Warning {data_dir}/gfem-efficiency.csv does not exist!")

    print("write-markdown-tables.py done!")

    return None

if __name__ == "__main__":
    main()