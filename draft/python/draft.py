import os
import re
import shutil
import argparse
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parse arguments !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_arguments():
    """
    """
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the command-line arguments
    parser.add_argument("--ms", type=str, required=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# check arguments !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_arguments(args, script):
    """
    """
    # Arguments
    ms = args.ms

    valid_args = {}

    # Check arguments and print
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Running {script} with:")

    if ms is not None:
        print(f"    manuscript: {ms}")

        valid_args["ms"] = ms

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return valid_args

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# copy assets !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_assets():
    """
    """
    # Get main directory paths
    main_dir = os.path.dirname(os.getcwd())

    # Copy assets from main directory
    print("Copying data from main directory ...")

    # Remove old directory if it exists
    if os.path.exists("assets/data"):
        shutil.rmtree("assets/data")

    shutil.copytree(os.path.join(main_dir, "assets/data"), "assets/data")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# copy figs !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def copy_figs(manuscript):
    """
    """
    # Get draft directory paths
    cwd = os.getcwd()
    fig_dir_draft = os.path.join(cwd, "assets/figs")

    # Get main directory paths
    main_dir = os.path.dirname(cwd)
    fig_dir_main = os.path.join(main_dir, "figs")

    # Create draft fig directory
    os.makedirs(fig_dir_draft, exist_ok=True)

    # Copy figures from main fig directory
    print("Copying figures:")

    # Read manuscript markdown file and search for .png
    with open(f"{manuscript}.md", "r") as md_file:
        for line in md_file:
            png_matches = re.findall(r"\(([^)]+\.png)\)", line)

            # Get filepaths of figures to copy from main directory
            for pngfile in png_matches:
                basename = os.path.basename(pngfile)
                fullpath = None

                for root, _, files in os.walk(fig_dir_main):
                    if basename in files:
                        fullpath = os.path.join(root, basename)
                        break

                if fullpath is None:
                    print(f"Warning: {basename} does not exist!")

                else:
                    shutil.copy(fullpath, os.path.join(fig_dir_draft, basename))
                    print(f"    {basename}")

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# replace placeholders !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def replace_placeholders(filepath, out_filepath):
    # Create a dictionary to cache file contents
    filenames = {}

    # Read the input markdown file
    with open(filepath, "r") as input_file:
        input_text = input_file.read()

    # Regular expression pattern to find {{ ... }} placeholders
    pattern = r'{{ ([^}]+) }}'

    # Function to replace placeholders with file contents
    def replace(match):
        file = match.group(1)
        replacement = ""

        if file in filenames:
            replacement = filenames[file]

        else:
            file_path = os.path.join("assets/pandoc", file)

            if os.path.exists(file_path):
                with open(file_path, "r") as file_contents:
                    replacement = file_contents.read()

                filenames[file] = replacement

            else:
                print(f"Warning: {file} does not exist!")

        return replacement

    # Perform replacements using regular expressions
    output_text = re.sub(pattern, replace, input_text)

    # Write the result to the output file
    with open(out_filepath, "w") as output_file:
        output_file.write(output_text)

    return None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# write markdown tables !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def write_markdown_tables():
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Pandoc dir
    pandoc_dir = "assets/pandoc"

    # CSV paths
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
        df_synth = df_synth[df_synth["SAMPLEID"].isin(["sm12000", "sm12127"])]
        df_synth.loc[df_synth["SAMPLEID"] == "sm12000", "SAMPLEID"] = "DSUM"
        df_synth.loc[df_synth["SAMPLEID"] == "sm12127", "SAMPLEID"] = "PSUM"

        # Combine data
        df_combined = pd.concat([df, df_synth], ignore_index=True).sort_values(by="SAMPLEID")

        # Rename columns
        col_headers = {"SAMPLEID": "Sample", "D_FRAC": "FI"}

        df_combined.rename(columns=col_headers, inplace=True)

        # Generate markdown table
        markdown_table = df_combined.to_markdown(index=False, floatfmt=".3g")

        # Table caption
        caption = (": Hypothetical upper mantle endmember compositions (in wt.%). "
                   "Fertility Index (FI) was calculated with a modal fractional melting "
                   "model based on TIO2 content (@eq:melt-fraction). "
                   "{#tbl:benchmark-samples}")

        # Write markdown table
        with open(f"{pandoc_dir}/benchmark-samples.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning: {df_path} does not exist!")

    if os.path.exists(f"{data_dir}/earthchem-counts.csv"):
        print("Writing earthchem-counts.md")

        # Earthchem imputed data counts
        df = pd.read_csv(f"{data_dir}/earthchem-counts.csv")

        # Subset oxides
        oxides = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "CR2O3", "H2O"]
        df = df[df["column"].isin(oxides)]

        # Rename columns
        df.loc[df["column"] == "FEO", "column"] = "FEOT"
        col_headers = {"column": "Oxide", "min": "Min", "max": "Max", "mean": "Mean",
                       "std": "Std", "median": "Median", "iqr": "IQR",
                       "measured": "Measured", "missing": "Missing", "imputed": "Imputed"}
        df.rename(columns=col_headers, inplace=True)

        # Generate markdown table
        markdown_table = df.to_markdown(index=False, floatfmt=".3g")

        # Table caption
        caption = (": Summary (in wt.%) of the filtered and standardized peridotite dataset "
                   "from Earthchem.org. {#tbl:earthchem-counts}")

        # Write markdown table
        with open(f"{pandoc_dir}/earthchem-counts.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning: {data_dir}/earthchem-counts.csv does not exist!")

    if os.path.exists(f"{data_dir}/rocmlm-config.csv"):
        print("Writing rocmlm-config.md")

        # MLM pro con
        df = pd.read_csv(f"{data_dir}/rocmlm-config.csv")

        # Rename columns
        col_headers = {"model": "Model", "hyperparam": "Hyperparameter",
                       "hyperparams_value": "Value", "hyperparams_tuned": "Tuned"}
        df.rename(columns=col_headers, inplace=True)

        # Generate markdown table
        markdown_table = df.to_markdown(index=False)

        # Table caption
        caption = (": RocMLM configuration. Hyperparameter values in parentheses are chosen "
                   "by a cross-validation grid search algorithm. All other hyperparameters "
                   "use defaults values (see regression model documentation on "
                   "[scikit-learn.org](htpps://scikit-learn.org)). {#tbl:rocmlm-config}")

        # Write markdown table
        with open(f"{pandoc_dir}/rocmlm-config.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning: {data_dir}/rocmlm-config.csv does not exist!")

    if os.path.exists(f"{data_dir}/rocmlm-performance.csv"):
        print("Writing rocmlm-performance.md")

        # RocMLM performance metrics
        df = pd.read_csv(f"{data_dir}/rocmlm-performance.csv")

        # Filter models
        df = df[df["sample"].isin(["SMAT128", "SMAT64", "SMAT32", "benchmark"])]

        # Transform units
        df["training_time_mean"] = df["training_time_mean"] * 1000
        df["inference_time_mean"] = df["inference_time_mean"] * 1000
        df["size"] = np.sqrt(df["size"])

        # Round Mb size
        df["model_size_mb"] = round(df["model_size_mb"], 3)
        df["model_size_mb"] = df["model_size_mb"].apply(lambda x: f"{x:.3g}")

        # Get X resolution
        def get_x_res(row):
            if row["sample"].startswith("SMA") and row["sample"][4:].isdigit():
                return int(row["sample"][4:])
            else:
                return 2

        df["x_res"] = df.apply(get_x_res, axis=1)

        # Rename sample
        def rename_sample(row):
            if row["sample"].startswith("SMA") and row["sample"][4:].isdigit():
                return row["sample"][:4]
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
#                    df[col] = df[col].astype(str) + " ± " + (df[std_col]).astype(str)

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

        # Table caption
        caption = (": RocMLM, size, efficiency, and performance measured on an (unseen) "
                   "validation dataset. {#tbl:rocmlm-performance}")

        # Write markdown table
        with open(f"{pandoc_dir}/rocmlm-performance.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

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

        # Table caption
        caption = (": Prediction efficiency (in seconds) for various bulk rock compositions. "
                   "{#tbl:gfem-efficiency}")

        # Write markdown table
        with open(f"{pandoc_dir}/gfem-efficiency.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning {data_dir}/gfem-efficiency.csv does not exist!")

    print("write-markdown-tables.py done!")

    return None