import os
import re
import shutil
import argparse
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

    if os.path.exists(f"{data_dir}/benchmark-samples-normalized.csv"):
        print("Writing benchmark-samples-normalized.md")

        # Benchmark compositions
        df = pd.read_csv(f"{data_dir}/benchmark-samples-normalized.csv")

        # Drop references
        df = df.drop(columns=["REFERENCE", "FE2O3", "H2O"])

        # Rename columns
        col_headers = {"SAMPLEID": "Sample", "SIO2": "SiO$_2$", "AL2O3": "Al$_2$O$_3$",
                       "CAO": "CaO", "MGO": "MgO", "FEO": "FeO", "K2O": "K$_2$O",
                       "NA2O": "Na$_2$O", "TIO2": "TiO$_2$", "CR2O3": "Cr$_2$O$_3$"}

        df.rename(columns=col_headers, inplace=True)

        # Generate markdown table
        markdown_table = df.to_markdown(index=False, floatfmt=".2f")

        # Table caption
        caption = (": Compositions (in wt. % oxides) for the benchmark samples: Primitive "
                   "Upper Mantle [PUM, @sun1989] and Depleted Morb Mantle "
                   "[DMM, @workman2005] {#tbl:benchmark-samples-normalized}")

        # Write markdown table
        with open(f"{pandoc_dir}/benchmark-samples-normalized.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning: {data_dir}/benchmark-samples-normalized.csv does not exist!")

    if os.path.exists(f"{data_dir}/mlm-pro-con.csv"):
        print("Writing mlm-pro-con.md")

        # MLM pro con
        df = pd.read_csv(f"{data_dir}/mlm-pro-con.csv")

        # Generate markdown table
        markdown_table = df.to_markdown(index=False)

        # Table caption
        caption = (": Advantages and disadvantages of various non-linear ML models. "
                   "{#tbl:mlm-pro-con}")

        # Write markdown table
        with open(f"{pandoc_dir}/mlm-pro-con.md", "w") as file:
            file.write(f"{markdown_table}\n")
            file.write(f"\n{caption}")

    else:
        print(f"Warning: {data_dir}/mlm-pro-con.csv does not exist!")

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
                    df[col] = df[col].astype(str) + " ± " + (df[std_col]).astype(str)

        # Drop the std_ columns
        df.drop([col for col in df.columns if "_std" in col], axis=1, inplace=True)

        # Drop program column
        df.drop(["program"], axis=1, inplace=True)

        # Select largest model size
        condition = df["size"] == df["size"].max()
        df = df[condition]

        # Drop size column
        df.drop(["size"], axis=1, inplace=True)

        # Rename columns
        df.columns = ["Model", "t$_{\\text{train}}$ (ms)",
                      "t$_{\\text{predict}}$ (ms)", "$\\epsilon_\\rho$ (g/cm$^3$)",
                      "$\\epsilon_{\\text{Vp}}$ (km/s)", "$\\epsilon_{\\text{Vs}}$ (km/s)",
                      "$\\epsilon_{\\text{melt}}$ (%)"]

        # Generate markdown table
        markdown_table = df.to_markdown(index=False)

        # Table caption
        caption = (": RocML performance measured on an independent (unseen) validation "
                   "dataset during kfold cross-validation (t = elapsed time; "
                   "$\\epsilon = RMSE$). Uncertainties are 2$\\sigma$ "
                   "{#tbl:rocml-performance}")

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