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
        df_synth = df_synth[df_synth["SAMPLEID"].isin(["sm12000", "sm23127"])]
        df_synth.loc[df_synth["SAMPLEID"] == "sm12000", "SAMPLEID"] = "DSUM"
        df_synth.loc[df_synth["SAMPLEID"] == "sm23127", "SAMPLEID"] = "PSUM"

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
                   "by a cross-validation grid search algorithm. Percentages refer to the "
                   "total number of training examples. All other hyperparameters use "
                   "defaults values (see regression model documentation on "
                   "[scikit-learn.org](htpps://scikit-learn.org)). {#tbl:rocmlm-config}")

        # Write markdown table
        with open(f"{pandoc_dir}/rocmlm-config.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning: {data_dir}/rocmlm-config.csv does not exist!")

    if os.path.exists(f"{data_dir}/rocml-performance.csv"):
        print("Writing rocml-performance.md")

        # RocML performance metrics
        df = pd.read_csv(f"{data_dir}/rocml-performance.csv")

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

        # Select two smallset model sizes (benchmark and random mixing models)
        two_smallest_sizes = np.sort(np.unique(df["size"]))[:2]
        df = df[df["size"].isin(two_smallest_sizes)]

        # Sort df
        df = df.sort_values(by=["size", "model"])

        # Rename columns
        df.columns = ["Model", "$n_\\text{train}$", "t$_\\text{train}$ (ms)",
                      "t$_\\text{pred}$ (ms)", "$\\varepsilon_{\\rho}$ (g/cm$^3$)",
                      "$\\varepsilon_{\\text{Vp}}$ (km/s)",
                      "$\\varepsilon_{\\text{Vs}}$ (km/s)"]

        # Generate markdown table
        markdown_table = df.to_markdown(index=False)

        # Table caption
        caption = (": RocML performance measured on an (unseen) validation dataset. "
                   "t = elapsed time, $\\varepsilon$ = RMSE. {#tbl:rocml-performance}")

        # Write markdown table
        with open(f"{pandoc_dir}/rocml-performance.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning: {data_dir}/rocml-performance.csv does not exist!")

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