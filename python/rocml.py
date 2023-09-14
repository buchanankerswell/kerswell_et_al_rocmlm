# Load packages
import os
import re
import sys
import ast
import yaml
import glob
import time
import math
import shutil
import zipfile
import fnmatch
import argparse
import warnings
import platform
import itertools
import subprocess
import numpy as np
import pandas as pd
import pkg_resources
from git import Repo
import urllib.request
import seaborn as sns
from scipy import stats
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from multiprocessing import Pool, cpu_count
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

# Print filepaths
def print_filepaths(folder_path):
    """
    Print the filepaths of all files in the specified folder path.

    Parameters:
        folder_path (str): The path of the folder whose filepaths need to be printed.

    Returns:
        None

    This function iterates through all items (files and subfolders) in the given folder_path
    and prints the absolute filepaths of all files found. It skips the special macOS file
    '.DS_Store' and only considers regular files (excluding subfolders).
    """
    for item in os.listdir(folder_path):
        if item == ".DS_Store":
             continue
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            print(f"    {item_path}")

# Extract info along a geotherm
def extract_info_geotherm(
        results,
        parameter,
        thermal_gradient=0.5,
        mantle_potential_T=1573,
        threshold=0.1):
    """
    Extracts relevant data points along a geotherm based on specified parameters.

    Parameters:
        results (dict): A dictionary containing the results of geothermal data.
                        It must have keys "P" and "T", representing pressure and temperature
                        values, respectively, and a key for the specified parameter.

        parameter (str): The name of the parameter for which data will be extracted along
                         the geotherm.

        thermal_gradient (float, optional): The thermal gradient used to calculate the
                                            geotherm.  Default is 0.5.

        mantle_potential_T (float, optional): The mantle potential temperature in Kelvin
                                              used to calculate the geotherm.
                                              Default is 1773 K.

        threshold (float, optional): The threshold used to determine data points' proximity
                                     to the geotherm.  Data points with pressure values
                                     within this threshold of the calculated geotherm_P
                                     will be extracted. Default is 0.1.

    Returns:
        tuple: A tuple containing three arrays: P_values, T_values, and parameter_values.
               - P_values (numpy.ndarray): An array of pressure values along the geotherm.
               - T_values (numpy.ndarray): An array of temperature values along the geotherm.
               - parameter_values (numpy.ndarray): An array of the specified parameter's
                 values along the geotherm.

    This function takes the input geothermal data, calculates the geotherm, and then
    extracts the relevant data points corresponding to the specified parameter along the
    geotherm. It performs unit conversions for pressure and temperature as necessary
    before calculations.
    """
    # Get PT and parameter values and transform units
    df = pd.DataFrame({
        "P": [P / 10 for P in results["P"]],
        "T": [T + 273 for T in results["T"]],
        parameter: results[parameter]
    }).sort_values(by="P")

    # Calculate geotherm
    df["geotherm_P"] = (df["T"] - mantle_potential_T) / (thermal_gradient * 35)

    # Subset df along geotherm
    df_geotherm = df[abs(df["P"] - df["geotherm_P"]) < threshold]

    # Extract the three vectors
    P_values = df_geotherm["P"].values
    T_values = df_geotherm["T"].values
    parameter_values = df_geotherm[parameter].values

    return P_values, T_values, parameter_values

# Append info from a dict to csv
def append_to_csv(file_path, data_dict):
    """
    Append data from a dictionary to a CSV file.

    Parameters:
        file_path (str): The path to the CSV file where data will be appended or created.

        data_dict (dict): A dictionary containing the data to be appended to the CSV file.
                          The keys represent column names, and the values are lists or arrays
                          containing the data to be appended.

    Returns:
        None

    This function checks if the specified CSV file exists. If it does not, it creates a new
    DataFrame using the data from the given data_dict and saves it as a new CSV file. If the
    CSV file already exists, it loads the existing DataFrame, appends the new data from the
    data_dict, and sorts the DataFrame by the "program" and "parameter" columns. Finally, it
    saves the updated DataFrame back to the CSV file.
    """
    # Check if the CSV file already exists
    if not pd.io.common.file_exists(file_path):
        df = pd.DataFrame(data_dict)
    else:
        df = pd.read_csv(file_path)

        # Append the new data dictionary to the DataFrame
        new_data = pd.DataFrame(data_dict)
        df = pd.concat([df, new_data], ignore_index=True)

    # Sort df
    df = df.sort_values(by=["sample", "model", "program", "grid"])

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

# Crop geotherms within PT bounds
def crop_geotherms(P_array, T_array):
    """
    Crop geotherms within PT bounds.

    This function takes two arrays representing pressure and temperature.

    The function then creates four geotherm arrays (geotherm1, geotherm2, and geotherm3) by
    linearly spacing temperature values between 273 K and T_max.

    The geotherm arrays are calculated using different temperature gradients.

    It proceeds to crop each geotherm array based on the PT bounds.
    Each geotherm is cropped to retain only the values that fall within the minimum and
    maximum pressure (P_min and P_max) and temperature (T_min and T_max) bounds obtained
    earlier.

    The resulting cropped geotherms and their corresponding temperature values are stored in
    separate lists, cropped_geotherms and cropped_T_values.

    Parameters:
        P_array (array-like): An array containing pressure values in some units.
        T_array (array-like): An array containing temperature values in Kelvin.

    Returns:
        tuple: A tuple containing two lists.
            - cropped_geotherms (list): A list of cropped geotherm arrays,
            where each array corresponds to a specific geotherm and contains temperature
            values that fall within the PT bounds.
            - cropped_T_values (list): A list of temperature arrays corresponding to the
            cropped geotherms, where each array contains temperature values that fall within
            the PT bounds.

    Note:
        - The function assumes that the units of pressure in P_array are consistent with
        those in the geotherm arrays. Ensure that the units are compatible for
        meaningful results.
        - The function relies on the NumPy library to perform array operations.
    """
    # Get min and max PT
    T_min, T_max = np.array(T_array).min(), np.array(T_array).max()
    P_min, P_max = np.array(P_array).min(), np.array(P_array).max()

    # Mantle potential temps
    T_mantle1 = 0 + 273
    T_mantle2 = 1500 + 273

    # Thermal gradients
    grad_mantle1 = 1
    grad_mantle2 = 0.5

    # Create geotherm arrays
    T1 = np.linspace(T_mantle1, T_max, num=100)
    T2 = np.linspace(T_mantle2, T_max, num=100)
    P1 = (T1 - T_mantle1) / (grad_mantle1 * 35)
    P2 = (T2 - T_mantle2) / (grad_mantle2 * 35)

    # Crop each geotherm array based on PT bounds
    P_arrays, T_arrays = [P1, P2], [T1, T2]

    cropped_P, cropped_T = [], []

    for i in range(len(P_arrays)):
        P, T = P_arrays[i], T_arrays[i]

        # Crop based on PT bounds
        mask = np.logical_and(P >= P_min, P <= P_max)
        P, T = P[mask], T[mask]

        mask = np.logical_and(T >= T_min, T <= T_max)
        P, T = P[mask], T[mask]

        # Append the cropped P to the list
        cropped_P.append(P)
        cropped_T.append(T)

    return cropped_P, cropped_T

# Layout plots horizontally
def combine_plots_horizontally(
        image1_path,
        image2_path,
        output_path,
        caption1,
        caption2,
        font_size=150,
        caption_margin=25,
        dpi=330):
    """
    Combine plots horizontally and add captions in the upper left corner.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        output_path (str): Path to save the combined image with captions.
        font_size (int, optional): Font size of the captions. Defaults to 100.
        caption_margin (int, optional): Margin between the captions and the images.
        dpi (int, optional): DPI (dots per inch) of the output image. Defaults to 330.
    """

    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum height between the two images
    max_height = max(image1.height, image2.height)
    max_width = max(image1.width, image2.width)

    # Create a new image with twice the width and the maximum height
    combined_image = Image.new("RGB", (max_width * 2, max_height), (255, 255, 255))

    # Set the DPI metadata
    combined_image.info["dpi"] = (dpi, dpi)

    # Paste the first image on the left
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right
    combined_image.paste(image2, (max_width, 0))

    # Add captions
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("Arial", font_size)
    caption_margin = caption_margin

    # Add caption "a"
    draw.text(
        (caption_margin, caption_margin),
        caption1,
        font=font,
        fill="black"
    )

    # Add caption "b"
    draw.text(
        (image1.width + caption_margin, caption_margin),
        caption2,
        font=font,
        fill="black"
    )

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))

# Layout plots vertically
def combine_plots_vertically(
        image1_path,
        image2_path,
        output_path,
        caption1,
        caption2,
        font_size=150,
        caption_margin=25,
        dpi=330):
    """
    Combine two plots vertically.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        output_path (str): Path to save the combined image.
    """
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum width between the two images
    max_width = max(image1.width, image2.width)

    # Create a new image with the maximum width and the sum of the heights
    combined_height = image1.height + image2.height
    combined_image = Image.new("RGB", (max_width, combined_height), (255, 255, 255))

    # Paste the first image on the top
    combined_image.paste(image1, (0, 0))

    # Paste the second image below the first
    combined_image.paste(image2, (0, image1.height))

    # Add captions
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("Arial", font_size)
    caption_margin = caption_margin

    # Add caption "a"
    draw.text(
        (caption_margin, caption_margin),
        caption1,
        font=font,
        fill="black"
    )

    # Add caption "b"
    draw.text(
        (caption_margin, image1.height + caption_margin),
        caption2,
        font=font,
        fill="black"
    )

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))

# Read conda packages from yaml
def get_conda_packages(conda_file):
    """
    Read conda packages from a YAML file.

    Parameters:
        conda_file (str): The path to the Conda YAML file.

    Returns:
        list: A list of Conda package dependencies specified in the YAML file.

    Raises:
        IOError: If there was an error reading the Conda file.
        yaml.YAMLError: If there was an error parsing the YAML file.

    """
    try:
        with open(conda_file, "r") as file:
            conda_data = yaml.safe_load(file)
        return conda_data.get("dependencies", [])
    except (IOError, yaml.YAMLError) as e:
        print(f"Error reading Conda file: {e}")
        return []

# Read makefile variables
def read_makefile_variable(makefile, variable):
    try:
        with open(makefile, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.strip().startswith(variable):
                    return line.split("=")[1].strip()
    except IOError as e:
        print(f"Error reading Makefile: {e}")
    return None

# Print session info for logging
def print_session_info(conda_file=None, makefile=None):
    """
    Print session information for logging.

    Parameters:
        conda_file (str, optional): The path to the Conda YAML file. Defaults to None.
        makefile (str, optional): The path to the Makefile. Defaults to None.

    """
    print("Session info:")

    # Print Python version
    python_version = sys.version_info
    version_string = ".".join(map(str, python_version))
    print(f"    Python Version: {version_string}")

    # Print package versions
    print("    Loaded packages:")
    if conda_file:
        conda_packages = get_conda_packages(conda_file)
        for package in conda_packages:
            if isinstance(package, str) and package != "python":
                package_name = package.split("=")[0]
                try:
                    version = pkg_resources.get_distribution(package_name).version
                    print(f"        {package_name} Version: {version}")
                except pkg_resources.DistributionNotFound:
                    print(f"        {package_name} not found ...")
    else:
        print("    No Conda file provided ...")

    # Print operating system information
    os_info = platform.platform()
    print(f"    Operating System: {os_info}")

    # Print random seed
    if makefile:
        seed = read_makefile_variable(makefile, "SEED")
        if seed:
            print(f"    Random Seed (from Makefile): {seed}")
        else:
            print("    SEED variable not found in Makefile ...")
    else:
        print("No Makefile provided.")

# Check non-matching strings
def check_non_matching_strings(list1, list2):
    """
    Check for non-matching strings between two lists.

    Parameters:
        list1 (list): The first list of strings.
        list2 (list): The second list of strings.

    Returns:
        bool: True if there are non-matching strings between the two lists, False otherwise.

    """
    set1 = set(list1)
    set2 = set(list2)
    non_matching_strings = set1 - set2

    return bool(non_matching_strings)

# Parse string argument as list of numbers
def parse_list_of_numbers(arg):
    """
    Parse a string argument as a list of numbers.

    Args:
        arg (str): The string argument to parse.

    Returns:
        list: A list of numbers parsed from the input string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid list of 11 numbers.

    """
    try:
        num_list = ast.literal_eval(arg)
        if (
            isinstance(num_list, list) and
            len(num_list) == 11 and
            all(isinstance(num, (int, float)) for num in num_list)
        ):
            return num_list
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid list: {arg} ...\nIt must contain exactly 11 numerical values ..."
            )
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid list: {arg} ...\nIt must contain exactly 11 numerical values ..."
        )

# Parse string argument as list of strings
def parse_list_of_strings(arg):
    """
    Parse a string argument as a list of strings.

    Args:
        arg (str): The string argument to parse.

    Returns:
        list: A list of strings parsed from the input string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a valid list of strings.

    """
    if arg == "all":
        return arg
    try:
        str_list = ast.literal_eval(arg)
        if (
            isinstance(str_list, list) and
            all(isinstance(item, str) for item in str_list)
        ):
            return str_list
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid list: {arg} ...\nIt must contain a valid list of strings ..."
            )
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid list: {arg} ...\nIt must contain a valid list of strings ..."
        )

# Parse arguments for python scripts
def parse_arguments():
    """
    Parse the command-line arguments for the build-database.py script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Raises:
        argparse.ArgumentTypeError: If any of the required arguments are missing.

    """
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the command-line arguments
    parser.add_argument(
        "--Pmin",
        type=int,
        help="Specify the Pmin argument ...",
        required=False
    )
    parser.add_argument(
        "--Pmax",
        type=int,
        help="Specify the Pmax argument ...",
        required=False
    )
    parser.add_argument(
        "--Tmin",
        type=int,
        help="Specify the Tmin argument ...",
        required=False
    )
    parser.add_argument(
        "--Tmax",
        type=int,
        help="Specify the Tmax argument ...",
        required=False
    )
    parser.add_argument(
        "--res",
        type=int,
        help="Specify the res argument ...",
        required=False
    )
    parser.add_argument(
        "--comp",
        type=parse_list_of_numbers,
        help="Specify the comp argument ...",
        required=False
    )
    parser.add_argument(
        "--frac",
        type=str,
        help="Specify the frac argument ...",
        required=False
    )
    parser.add_argument(
        "--sampleid",
        type=str,
        help="Specify the sampleid argument ...",
        required=False
    )
    parser.add_argument(
        "--params",
        type=parse_list_of_strings,
        help="Specify the params argument ...",
        required=False
    )
    parser.add_argument(
        "--models",
        type=parse_list_of_strings,
        help="Specify the models argument ...",
        required=False
    )
    parser.add_argument(
        "--normox",
        type=parse_list_of_strings,
        help="Specify the normox argument ...",
        required=False
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Specify the source argument ...",
        required=False
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Specify the strategy argument ...",
        required=False
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Specify the n argument ...",
        required=False
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Specify the k argument ...",
        required=False
    )
    parser.add_argument(
        "--parallel",
        type=str,
        help="Specify the parallel argument ...",
        required=False
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        help="Specify the nprocs argument ...",
        required=False
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        help="Specify the kfolds argument ...",
        required=False
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Specify the seed argument ...",
        required=False
    )
    parser.add_argument(
        "--colormap",
        type=str,
        help="Specify the colormap argument ...",
        required=False
    )
    parser.add_argument(
        "--figdir",
        type=str,
        help="Specify the figdir argument ...",
        required=False
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Specify the outdir argument ...",
        required=False
    )
    parser.add_argument(
        "--datadir",
        type=str,
        help="Specify the datadir argument ...",
        required=False
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specify the dataset argument ...",
        required=False
    )
    parser.add_argument(
        "--tune",
        type=str,
        help="Specify the tune argument ...",
        required=False
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the parallel argument to a boolean value based on the string value
    parallel = args.parallel.lower() == "true" if args.parallel else False

    # Assign the modified parallel value back to the args object
    args.parallel = parallel

    return args

# Check arguments
def check_arguments(args, script):
    """
    Validate and print the arguments for a specific script.

    Parameters:
        args (argparse.Namespace): The argparse.Namespace object containing parsed
                                   command-line arguments.

        script (str): The name of the script being executed.

    Returns:
        dict: A dictionary containing the valid arguments with their corresponding values.

    This function takes the argparse.Namespace object 'args', which contains parsed
    command-line arguments, and the 'script' name for which the arguments are being checked.
    It validates the arguments one by one, prints their values, and stores them in a
    dictionary of valid_args. If any argument is invalid, it raises a ValueError with the
    corresponding error message.
    """
    # Arguments
    Pmin = args.Pmin
    Pmax = args.Pmax
    Tmin = args.Tmin
    Tmax = args.Tmax
    res = args.res
    dataset = args.dataset
    comp = args.comp
    frac = args.frac
    sampleid = args.sampleid
    params = args.params
    models = args.models
    tune = args.tune
    normox = args.normox
    source = args.source
    strategy = args.strategy
    n = args.n
    k = args.k
    parallel = args.parallel
    nprocs = args.nprocs
    kfolds = args.kfolds
    seed = args.seed
    colormap = args.colormap
    figdir = args.figdir
    outdir = args.outdir
    datadir = args.datadir

    # MAGEMin oxide options
    oxide_list_magemin = [
        "SiO2",
        "Al2O3",
        "CaO",
        "MgO",
        "FeO",
        "K2O",
        "Na2O",
        "TiO2",
        "Fe2O3",
        "Cr2O3",
        "H2O"

    ]

    # Benchmark sample ids
    benchmark_sample_ids = ["PUM", "DMM", "RE46", "NMORB"]

    # Benchmark sample data
    datafile = "assets/data/benchmark-samples.csv"

    valid_args = {}

    # Check arguments and print
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Running {script} with:")

    if Pmin is not None:
        print(f"    Pmin: {Pmin}")

        valid_args["Pmin"] = Pmin

    if Pmax is not None:
        print(f"    Pmax: {Pmax}")

        valid_args["Pmax"] = Pmax

    if Tmin is not None:
        print(f"    Tmin: {Tmin}")

        valid_args["Tmin"] = Tmin

    if Tmax is not None:
        print(f"    Tmax: {Tmax}")

        valid_args["Tmax"] = Tmax

    if res is not None:
        if res > 128:
            raise ValueError(
                "Invalid --res argument ...\n"
                "--res must be <= 128"
            )

        if Tmin is not None and Tmax is not None:
            print(f"    Trange: [{Tmin}, {Tmax}, {res}]")

        elif Pmin is not None and Pmax is not None:
            print(f"    Prange: [{Pmin}, {Pmax}, {res}]")

        else:
            print(f"    training dataset resolution: {res}")

        valid_args["res"] = res

    if comp is not None:
        print(f"    comp: {comp}")

        # Sample composition
        sample_comp = comp

        valid_args["comp"] = comp

    if frac is not None:
        if frac not in ["mol", "wt"]:
            raise ValueError(
                "Invalid --frac argument ...\n"
                "Use --frac=mol or --frac=wt"
            )

        print(f"    frac: {frac}")

        valid_args["frac"] = frac

    if sampleid is not None:
        if sampleid in benchmark_sample_ids:
            # Get sample composition
            sample_comp = get_benchmark_sample_for_magemin(datafile, sampleid)

        print(f"    sampleid: {sampleid}")

        valid_args["sampleid"] = sampleid

    if params is not None:
        print("    targets:")

        for param in params:
            print(f"        {param}")

        valid_args["params"] = params

    if models is not None:
        print("    ML models:")

        for model in models:
            print(f"        {model}")

        valid_args["models"] = models

    if tune is not None:
        print(f"    tune: {tune}")

        if tune == "False":
            tune = False
        else:
            tune = True

        valid_args["tune"] = tune

    if colormap is not None:
        if colormap not in ["viridis", "bone", "pink", "seismic", "grey", "blues"]:
            raise ValueError(
                "Invalid --colormap argument ...\n"
                f"Use --colormap=viridis or bone or pink or seismic or grey or blues"
            )

        print(f"    colormap: {colormap}")

        valid_args["colormap"] = colormap

    if normox is not None:
        if normox != "all":
            if check_non_matching_strings(normox, oxide_list_magemin):
                raise ValueError(
                    "Invalid --normox argument ...\n"
                    f"Can only normalize to oxides {oxide_list_magemin}"
                )

            print(f"    Normalizing composition to:")

            for oxide in normox:
                print(f"        {oxide}")

        # Normalize composition
        sample_norm = normalize_sample(sample=sample_comp, components=normox)

        print("    Normalized composition:")

        for component, value in zip(oxide_list_magemin, sample_norm):
            formatted_value = "{:.3f}".format(value)
            print(f"        {component}: {formatted_value}")


        valid_args["normox"] = normox

    if source is not None:
        if source not in ["sample", "earthchem"]:
            raise ValueError(
                "Invalid --source argument ...\n"
                "Use --source=earthchem or --source=sample"
            )

        print(f"    source: {source}")

        valid_args["source"] = source

    if strategy is not None:
        if strategy not in ["random", "batch"]:
            raise ValueError(
                "Invalid --strategy argument ...\n"
                "Use --source=earthchem or --source=sample"
            )

        print(f"    strategy: {strategy}")

        valid_args["strategy"] = strategy

    if n is not None:
        print(f"    n: {n}")

        valid_args["n"] = n

    if k is not None:
        print(f"    k: {k}")

        valid_args["k"] = k

    if parallel is not None:
        if not isinstance(parallel, bool):
            raise ValueError(
                "Invalid --parallel argument ...\n"
                "--parallel must be either True or False"
            )

        print(f"    parallel: {parallel}")

        valid_args["parallel"] = parallel

    if nprocs is not None:
        if nprocs > os.cpu_count():
            raise ValueError(
                "Invalid --nprocs argument ...\n"
                f"--nprocs cannot be greater than cores on system ({os.cpu_count}) ..."
            )

        print(f"    nprocs: {nprocs}")

        valid_args["nprocs"] = nprocs

    if seed is not None:
        print(f"    seed: {seed}")

        valid_args["seed"] = seed

    if kfolds is not None:
        print(f"    kfolds: {kfolds}")

        valid_args["kfolds"] = kfolds

    if outdir is not None:
        if len(outdir) > 55:
            raise ValueError(
                "Invalid --outdir argument ...\n"
                f"--outdir cannot be greater than 55 characters ..."
                f"{outdir}"
            )

        print(f"    outdir: {outdir}")

        valid_args["outdir"] = outdir

    if figdir is not None:
        print(f"    figdir: {figdir}")

        valid_args["figdir"] = figdir

    if datadir is not None:
        print(f"    datadir: {datadir}")

        valid_args["datadir"] = datadir

    if dataset is not None:
        print(f"    dataset: {dataset}")

        valid_args["dataset"] = dataset

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return valid_args

# Download data from repo
def download_and_unzip(url, destination):
    """
    Download a zip file from a given URL and extract its contents to a specified destination.

    Args:
        url (str): The URL of the zip file to download.
        destination (str): The folder where the contents of the zip file will be extracted.

    Raises:
        urllib.error.HTTPError: If there is an error accessing the URL.

    """
    # Download the file
    urllib.request.urlretrieve(url, "assets.zip")

    # Extract the contents of the zip file
    with zipfile.ZipFile("assets.zip", "r") as zip_ref:
        zip_ref.extractall(destination)

    # Remove the zip file
    os.remove("assets.zip")

# Read csv files
def read_geochemical_data(file_path):
    """
    Reads a CSV file containing geochemical data and returns the data as a pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The geochemical data read from the CSV file.
    """

    data = pd.read_csv(file_path)
    return data

# Sample batches from the earthchem database
def batch_sample_for_magemin(datafile, batch_size=1, k=0):
    """
    Splits the data from the given datafile into batches of size `batch_size`
    and returns the k-th batch.

    Args:
        datafile (str): Path to the CSV data file.
        batch_size (int): Size of each batch (default: 1).
        k (int): Index of the batch to retrieve (default: 0).

    Returns:
        tuple: Two lists, the first containing sample_ids and the second containing
        lists of oxide compositions for the k-th batch.

    """
    # All oxides needed for MAGEMin
    component_list = [
        "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O",
        "Na2O", "TiO2", "Fe2O3", "Cr2O3", "H2O"
    ]

    # Make uppercase
    oxides = [oxide.upper() for oxide in component_list]

    # Read the datafile in chunks of size batch_size
    df_iterator = pd.read_csv(datafile, chunksize=batch_size)

    # Initialize variables for the k-th batch
    sample_ids = []
    compositions = []

    # Iterate until the k-th batch
    for i, chunk in enumerate(df_iterator):
        if i == k:
            # Process the k-th batch
            for _, row in chunk.iterrows():
                sample_ids.append(row["SAMPLE ID"])
                composition = []
                for oxide in oxides:
                    if oxide in row.index and pd.notnull(row[oxide]):
                        composition.append(float(row[oxide]))
                    else:
                        composition.append(0.01)
                compositions.append(composition)
            break

    return sample_ids, compositions

# Get composition of a single benchmark mantle samples
def get_benchmark_sample_for_magemin(datafile, sample_id):
    """
    Extracts the oxide compositions needed for the "MAGEMin" analysis
    for a single sample specified by name from the given datafile.

    Args:
        datafile (str): Path to the CSV data file.
        sample_id (str): Name of the sample to select.

    Returns:
        list: Oxide compositions for the specified sample.

    """
    # All oxides needed for MAGEMin
    component_list = [
        "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O",
        "Na2O", "TiO2", "Fe2O3", "Cr2O3", "H2O"
    ]

    # Make uppercase
    oxides = [oxide.upper() for oxide in component_list]

    # Read the data file
    df = pd.read_csv(datafile)

    # Subset the DataFrame based on the sample name
    subset_df = df[df["NAME"] == sample_id]

    if subset_df.empty:
        raise ValueError("Sample name not found in the dataset ...")

    # Get the oxide compositions for the selected sample
    composition = []
    for oxide in oxides:
        if oxide in subset_df.columns and pd.notnull(subset_df[oxide].iloc[0]):
            composition.append(float(subset_df[oxide].iloc[0]))
        else:
            composition.append(0.01)

    return composition

# Sample randomly from the earthchem database
def random_sample_for_magemin(datafile, n=1, seed=None):
    """
    Randomly samples n rows from the given datafile and
    extracts the oxide compositions needed for the "MAGEMin" analysis.

    Args:
        datafile (str): Path to the CSV data file.
        n (int): Number of random samples to extract (default: 1).
        seed (int): Random seed for reproducibility (default: None).

    Returns:
        tuple: Two lists, the first containing sample_ids and the second containing
        lists of oxide compositions for each random sample.

    """
    # All oxides needed for MAGEMin
    component_list = [
        "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O",
        "Na2O", "TiO2", "Fe2O3", "Cr2O3", "H2O"
    ]

    # Make uppercase
    oxides = [oxide.upper() for oxide in component_list]

    # Random sampling
    df = pd.read_csv(datafile)
    random_rows = df.sample(n, random_state=seed)

    # Get sample names and oxides in correct order for each random sample
    sample_ids = []
    compositions = []

    for _, random_row in random_rows.iterrows():
        sample_ids.append(random_row["SAMPLE ID"])
        composition = []
        for oxide in oxides:
            if oxide in random_row.index and pd.notnull(random_row[oxide]):
                composition.append(float(random_row[oxide]))
            else:
                composition.append(0.01)
        compositions.append(composition)

    return sample_ids, compositions

# Normalize components
def normalize_sample(sample, components="all"):
    """
    Normalize the concentrations for a subset of components.

    Args:
        sample (list): List of concentrations representing the components.
        components (list): List of components to normalize the concentrations for.

    Returns:
        list: Normalized concentrations for each component in the same order.

    Raises:
        ValueError: If the input sample list does not have exactly 11 components.

    """
    # No normalizing for all components
    if components == "all":
        return sample

    # MAGEMin req components
    component_list = [
        "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O",
        "Na2O", "TiO2", "Fe2O3", "Cr2O3", "H2O"
    ]

    # Check input
    if len(sample) != 11:
        error_message = (
            f"The input sample list must have exactly 11 components ...\n" +
            f"{component_list}"
        )
        raise ValueError(error_message)

    # Filter components
    subset_sample = [
        c if comp in components else 0.01 for c, comp in zip(sample, component_list)
    ]

    # Normalize
    total_subset_concentration = sum([c for c in subset_sample if c != 0.01])
    normalized_concentrations = []

    for c, comp in zip(sample, component_list):
        if comp in components:
            normalized_concentration = (
                (c / total_subset_concentration) * 100 if c != 0.01 else 0.01
            )
        else:
            normalized_concentration = 0.01
        normalized_concentrations.append(normalized_concentration)

    return normalized_concentrations

# Download github repo as submodule
def download_github_submodule(repository_url, submodule_dir):
    """
    Download a GitHub repository as a submodule.

    Args:
        repository_url (str): The URL of the GitHub repository.
        submodule_dir (str): The directory where the submodule will be cloned.

    Raises:
        Exception: If an error occurs while cloning the GitHub repository.

    Note:
        This function checks if the submodule directory already exists and deletes it
        before cloning the submodule and recursively cloning its contents.
    """
    # Check if submodule directory already exists and delete it
    if os.path.exists(submodule_dir):
        shutil.rmtree(submodule_dir)

    # Clone submodule and recurse its contents
    try:
        repo = Repo.clone_from(repository_url, submodule_dir, recursive=True)
    except Exception as e:
        print(f"An error occurred while cloning the GitHub repository: {e} ...")

# Count lines in a file
def count_lines(filename):
    """
    Count the number of lines in a file.

    Args:
        filename (str): The path to the file.

    Returns:
        int: The number of lines in the file.
    """
    line_count = 0
    with open(filename, "r") as file:
        for line in file:
            line_count += 1
    return line_count

# Merge dictionaries
def merge_dictionaries(dictionaries):
    """
    Merge a list of dictionaries by key terms.

    Args:
        dictionaries (list): A list of dictionaries to merge.

    Returns:
        dict: A single dictionary with merged key-value pairs.

    Raises:
        ValueError: If the dictionaries list is empty.
    """
    if not dictionaries:
        raise ValueError("The dictionaries list is empty.")

    merged_dict = {}
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if key in merged_dict:
                if isinstance(merged_dict[key], list):
                    if isinstance(value, list):
                        merged_dict[key].extend(value)
                    else:
                        merged_dict[key].append(value)
                else:
                    if isinstance(value, list):
                        merged_dict[key] = [merged_dict[key]] + value
                    else:
                        merged_dict[key] = [merged_dict[key], value]
            else:
                merged_dict[key] = value

    return merged_dict

# Create MAGEMin input
def create_magemin_input(
        P_range,
        T_range,
        composition,
        mode=0,
        run_name="test",
        out_dir="runs"):
    """
    Creates an input string for MAGEMin.

    Args:
        P_range (list): Start, end, and step values for the pressure range.
        T_range (list): Start, end, and step values for the temperature range.
        composition (list): Bulk composition with the following oxides:
            [SiO2, Al2O3, CaO, MgO, FeOt, K2O, Na2O, TiO2, O, Cr2O3, H2O]
        mode (int, optional): The mode value for MAGEMin. Default is 0.
        run_name (str, optional): The name of the MAGEMin run. Default is "test".

    Returns:
        None

    Prints:
        - Information about the created MAGEMin input file.
        - Pressure range and temperature range.
        - Composition [wt%] of the oxides.

    Creates:
        - A directory "runs" if it does not exist.
        - Writes the input string for MAGEMin to a file named "run_name.dat"
        inside the "runs" directory.
    """
    # Create directory
    directory = f"{os.getcwd()}/{out_dir}"
    os.makedirs(directory, exist_ok=True)

    # Setup PT vectors
    magemin_input = ""
    pressure_values = np.arange(
        float(P_range[0]),
        float(P_range[1]) + float(P_range[2]),
        float(P_range[2])
    )
    temperature_values = np.arange(
        float(T_range[0]),
        float(T_range[1]) + float(T_range[2]),
        float(T_range[2])
    )

    # Expand PT vectors into grid
    combinations = list(itertools.product(pressure_values, temperature_values))
    for p, t in combinations:
        magemin_input += (
            f"{mode} {p} {t} {composition[0]} {composition[1]} {composition[2]} "
            f"{composition[3]} {composition[4]} {composition[5]} {composition[6]} "
            f"{composition[7]} {composition[8]} {composition[9]} {composition[10]}\n"
        )

    # Write input file
    with open(f"{directory}/{run_name}.dat", "w") as f:
        f.write(magemin_input)

# Move files from MAGEMin output dir
def cleanup_ouput_dir(run_name, out_dir="runs"):
    """
    Move files from the MAGEMin output directory to a new directory based on the run name.
    Also moves the input data file into the new directory and removes the output directory.

    Args:
        run_name (str): The name of the run, used to create the new directory and rename
            the files.

    Returns:
        None
    """
    # Create the directory based on the run_name
    directory = f"{os.getcwd()}/{out_dir}/{run_name}"
    os.makedirs(directory, exist_ok=True)

    # Get a list of all files in the "output" directory matching the pattern
    files = os.listdir("output")
    matching_files = [file for file in files if file.startswith("_pseudosection")]

    # Rename and move the files
    for file in matching_files:
        new_filename = f"_{run_name}{file[len('_pseudosection'):]}"
        new_filepath = os.path.join(directory, new_filename)
        old_filepath = os.path.join("output", file)
        shutil.move(old_filepath, new_filepath)

    # Move input data file into directory
    if os.path.isfile(f"{directory}/{run_name}.dat"):
        os.remove(f"{directory}/{run_name}.dat")
        shutil.move(f"{directory}.dat", directory)
    else:
        shutil.move(f"{directory}.dat", directory)

    # Remove MAGEMin output directory
    shutil.rmtree("output")

# Run MAGEMin
def run_magemin(
        program_path=None,
        run_name="test",
        comp_type="wt",
        database="ig",
        parallel=True,
        nprocs=None,
        out_dir="runs"):
    """
    Runs MAGEMin with the specified parameters.

    Args:
        program_path (str, optional): The location of the MAGEMin executable.
        run_name (str, optional): The name of the MAGEMin run. Default is "test".
        mode (int, optional): The mode value for MAGEMin. Default is 0.
        comp_type (str, optional): The composition type for MAGEMin. Default is "wt".
        database (str, optional): The database for MAGEMin. Default is "ig".
        parallel (bool, optional): Determines whether to run MAGEMin in parallel.
        Default is True.
        nprocs (int, optional): The number of processes to use in parallel execution.
        Default is os.cpu_count()-2.
        verbose (bool, optional): Determines whether to print verbose output.

    Returns:
        None

    Prints:
        - Information about the execution of MAGEMin, including the command executed
          and elapsed time.
        - Prints the elapsed time in seconds.

    Calls:
        - cleanup_output_dir(run_name): Moves the output files and cleans up the
          directory after MAGEMin execution.
    """
    # Check for MAGEMin program
    if program_path is None:
        sys.exit("Please provide location to MAGEMin executable ...")

    # Count number of pt points to model with MAGEMin
    input_path = f"{os.getcwd()}/{out_dir}/{run_name}.dat"
    n_points = count_lines(input_path)

    # Check for input MAGEMin input files
    if not os.path.exists(input_path):
        sys.exit("No MAGEMin input files to run ...")

    # Execute MAGEMin in parallel with MPI
    if parallel == True:
        if nprocs > os.cpu_count():
            raise ValueError(
                f"Number of processors {os.cpu_count()} is less than nprocs argument ...\n"
                "Choose fewer nprocs ..."
            )
        elif nprocs is not None and nprocs < os.cpu_count():
            nprocs = nprocs
        elif nprocs is None:
            nprocs = os.cpu_count()-2
        exec = (
            f"mpirun -np {nprocs} {program_path}MAGEMin --File={input_path} "
            f"--n_points={n_points} --sys_in={comp_type} --db={database}"
        )
    # Or execute MAGEMin in sequence
    else:
        exec = (
            f"{program_path}MAGEMin --File={input_path} "
            f"--n_points={n_points} --sys_in={comp_type} --db={database}"
        )

    # Run MAGEMin
    shell_process = subprocess.run(exec, shell=True, text=True)

    # Move output files and cleanup directory
    cleanup_ouput_dir(run_name, out_dir)

# Read pseudosection info from MAGEMin output file
def read_magemin_pseudosection_data(filename):
    """
    Read and process pseudosection data from a file.

    Args:
        filename (str): The path to the file containing the pseudosection data.

    Returns:
        dict: A dictionary containing the processed pseudosection data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If there is an error in parsing the data.
    """
    # Open file
    with open(filename, "r") as file:
        lines = file.readlines()

    # Skip the comment line
    lines = lines[1:]

    # Extract results
    results = []
    i_point = 0
    while lines:
        # Read line with PT, Gamma, etc.
        line = lines.pop(0)
        a = list(map(float, line.split()))
        num_point = int(a[0])
        status = int(a[1])
        p = a[2]
        t = a[3]
        gibbs = a[4]
        br_norm = a[5]
        gamma = a[6:17]
        vp = a[17]
        vs = a[18]
        entropy = a[19]

        # Get stable solutions and endmember info
        stable_solutions = []
        stable_fractions = []
        density = []
        compositional_var = []
        em_fractions = []
        em_list = []

        i = 0
        line = lines.pop(0)
        while line.strip():
            out = line.split()
            data = []
            for value in out:
                try:
                    data.append(float(value))
                except ValueError:
                    data.append(value)

            stable_solutions.append(data[0])
            stable_fractions.append(data[1])
            density.append(data[2])

            comp_var = []
            em_frac = []
            em = []
            if len(data) > 4:
                n_xeos = int(data[3])
                comp_var = data[4:4 + n_xeos]
                em = out[4 + n_xeos::2]
                em_frac = data[5 + n_xeos::2]

            compositional_var.append(comp_var)
            em_fractions.append(em_frac)
            em_list.append(em)

            i += 1
            line = lines.pop(0)

        # Extract melt fraction
        ind_liq = [idx for idx, sol in enumerate(stable_solutions) if sol == "liq"]
        if ind_liq:
            liq = stable_fractions[ind_liq[0]]
        else:
            liq = 0

        # Compute average density of full assemblage
        density_total = sum(frac * dens for frac, dens in zip(stable_fractions, density))

        # Compute density of liq
        if liq > 0:
            density_liq = density[ind_liq[0]]
            ind_sol = [idx for idx, sol in enumerate(stable_solutions) if sol != "liq"]
            if ind_sol:
                density_sol =\
                    sum(
                        frac * dens for frac, dens in zip(
                            [stable_fractions[idx] for idx in ind_sol],
                            [density[idx] for idx in ind_sol]
                        )
                    ) / sum([stable_fractions[idx] for idx in ind_sol])
            else:
                density_sol = 0
            density_mix = (liq * density_liq + (1 - liq) * density_sol)
        else:
            density_liq = 0
            density_sol = 0
            density_mix = 0

        # Append results to the list
        results.append({
            "Point": num_point,
            "Status": status,
            "P": p,
            "T": t,
            "Gibbs": gibbs,
            "BrNorm": br_norm,
            "Gamma": [gamma],
            "Vp": vp,
            "Vs": vs,
            "Entropy": entropy,
            "StableSolutions": [stable_solutions],
            "StableFractions": [stable_fractions],
            "Density": density,
            "CompositionalVar": [compositional_var],
            "EMList": [em_list],
            "EMFractions": [em_fractions],
            "LiquidFraction": liq,
            "DensityOfFullAssemblage": density_total,
            "DensityOfLiquid": density_liq,
            "DensityOfSolid": density_sol,
            "DensityOfMixture": density_mix
        })

        i_point += 1

    return results

# Process all MAGEMin output files in a directory
def process_magemin_grid(sample_id, dataset, res, out_dir="runs"):
    """
    Process multiple MAGEMin output files in a directory based on a filename pattern.

    Args:
        sample_id (str): The name of the MAGEMin run.

    Returns:
        dict: A single dictionary with merged key-value pairs from all files.

    Raises:
        FileNotFoundError: If the "runs" directory or the specific MAGEMin run
        directory does not exist.
    """
    # Check for MAGEMin output files
    if not os.path.exists(f"{os.getcwd()}/{out_dir}"):
        sys.exit("No MAGEMin output files to process ...")
    if not os.path.exists(f"{os.getcwd()}/{out_dir}/{sample_id}/magemin_{dataset}_{res}"):
        sys.exit("No MAGEMin output files to process ...")

    # Get filenames directory for files
    directory = f"{os.getcwd()}/{out_dir}/{sample_id}/magemin_{dataset}_{res}"
    pattern = f"_{sample_id}*.txt"

    results = []

    # Process files
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(root, filename)
            file_results = read_magemin_pseudosection_data(filepath)
            results.extend(file_results)

    merged_results = merge_dictionaries(results)

    # Compute assemblage variance (number of phases)
    assemblages = merged_results.get("StableSolutions")

    variance = []

    for assemblage in assemblages:
        unique_phases = set(assemblage)
        count = len(unique_phases)
        variance.append(count)

    merged_results["StableVariance"] = variance

    return merged_results

# Encode stable phase assemblage for plotting with imshow
def encode_phases(phases, filename=None):
    """
    Encode unique phase assemblages and their corresponding numbers.

    Args:
        phases (list): List of phase assemblages represented as lists of phases.

    Returns:
        tuple: A tuple containing two elements:
            - encoded_assemblages (list): List of encoded phase assemblages.
            - unique_assemblages (dict): Dictionary mapping unique phase assemblages
            to their encoded numbers.
    """
    unique_assemblages = {}
    encoded_assemblages = []

    # Encoding unique phase assemblages
    for phase_assemblage in phases:
        assemblage_tuple = tuple(sorted(phase_assemblage))
        if assemblage_tuple not in unique_assemblages:
            unique_assemblages[assemblage_tuple] = len(unique_assemblages) + 1

    # Save list of unique phase assemblages
    if filename is not None:
        df = pd.DataFrame(list(unique_assemblages.items()), columns=["Assemblage", "Index"])
        df = df[["Index", "Assemblage"]]
        df["Assemblage"] = df["Assemblage"].apply(" ".join)
        df.to_csv(filename, index=False)

    # Encoding phase assemblage numbers
    for phase_assemblage in phases:
        if phase_assemblage == "":
            encoded_assemblages.append(math.nan)
        else:
            encoded_assemblage = unique_assemblages[tuple(sorted(phase_assemblage))]
            encoded_assemblages.append(encoded_assemblage)

    return encoded_assemblages, unique_assemblages

# Get perplex grids
def process_perplex_grid(file_path_grid, file_path_assemblages):
    """
    Process the Perple_X assemblage file and extract the phase assemblages.

    Args:
        file_path_assemblages (str): The path to the Perple_X assemblage file.

    Returns:
        list: A list of phase assemblages, where each assemblage is represented
              as a list of phase names.

    Raises:
        FileNotFoundError: If the specified assemblage file does not exist.
        ValueError: If there is an error in parsing the assemblage data.
    """
    results = {
        "T": [],
        "P": [],
        "DensityOfFullAssemblage": [],
        "Vp": [],
        "Vs": [],
        "Entropy": [],
        "AssemblageIndex": [],
        "LiquidFraction": [],
        "StableSolutions": [],
        "StableVariance": []
    }

    with open(file_path_grid, "r") as file:
        # Skip lines until column headers are found
        for line in file:
            if line.strip().startswith("T(K)"):
                break

        # Read the data
        for line in file:
            values = line.split()
            if len(values) >= 8:
                try:
                    for i in range(8):
                        value = (
                            float(values[i])
                            if not math.isnan(float(values[i]))
                            else math.nan
                        )
                        if i == 0:  # "T" column
                            value -= 273  # Subtract 273 from "T"
                        if i == 1:  # "P" column
                            value /= 1000  # Divide "P" by 1000
                        if i == 6:  # "AssemblageIndex" column
                            value = (
                                int(value)
                                if not math.isnan(value)
                                else math.nan
                            )
                        if i == 7:  # "LiquidFraction" column
                            value /= 100  # Divide "LiquidFraction" by 100
                        results[list(results.keys())[i]].append(value)
                except ValueError:
                    continue

    # Get assemblages from file
    assemblages = process_perplex_assemblage(file_path_assemblages)

    # Add assemblage to dictionary
    for index in results.get("AssemblageIndex"):
        if math.isnan(index):
            results["StableSolutions"].append("")
        else:
            phases = assemblages[index]
            results["StableSolutions"].append(phases)

    # Add assemblage variance to dictionary
    for assemblage in results.get("StableSolutions"):
        if assemblage is None:
            count = math.nan
        else:
            unique_phases = set(assemblage)
            count = len(unique_phases)
        results["StableVariance"].append(count)

    return results

# Get phase assemblages from perplex fields
def process_perplex_assemblage(file_path):
    """
    Process the Perple_X assemblage file and extract the phase assemblages.

    Args:
        file_path (str): The path to the Perple_X assemblage file.

    Returns:
        dict: A dictionary where the keys represent line numbers in the file and
              the values are lists of phase names that form the corresponding
              phase assemblages.

    Raises:
        FileNotFoundError: If the specified assemblage file does not exist.
    """
    phase_dict = {}
    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            phases = line.split("-")[1].strip().split()
            cleaned_phases = [phase.split("(")[0].lower() for phase in phases]
            phase_dict[line_number] = cleaned_phases
    return phase_dict

# Transform results into 2D numpy array across PT space
def create_PT_grid(P, T, parameter_values, mask=False):
    """
    Create a 2D NumPy array representing a grid of parameter values across PT space.

    Args:
        P (list or array-like): A 1D array or list of P values.
        T (list or array-like): A 1D array or list of T values.
        parameter_values (list or array-like): A 1D array or list of parameter values.

    Returns:
        numpy.ndarray: A 2D NumPy array representing the grid of parameter values.
            The grid is created by reshaping the parameter values based on unique
            P and T values. Missing values in the grid are represented by NaN.
    """
    # Convert P, T, and parameter_values to arrays
    P = np.array(P)
    T = np.array(T)
    parameter_values = np.array(parameter_values)

    # Get unique P and T values
    unique_P = np.unique(P)
    unique_T = np.unique(T)

    if mask:
        # Define T range
        T_min, T_max = min(unique_T), max(unique_T)

        # Define P range
        P_min, P_max = min(unique_P), max(unique_P)

        # Mantle potential temps
        T_mantle1 = 0 + 273
        T_mantle2 = 1500 + 273

        # Thermal gradients
        grad_mantle1 = 1
        grad_mantle2 = 0.5

        # Calculate mantle geotherms
        geotherm1 = (unique_T - T_mantle1) / (grad_mantle1 * 35)
        geotherm2 = (unique_T - T_mantle2) / (grad_mantle2 * 35)

        # Find boundaries
        T1_Pmax = (P_max * grad_mantle1 * 35) + T_mantle1
        P1_Tmin = (T_min - T_mantle1) / (grad_mantle1 * 35)
        T2_Pmin = (P_min * grad_mantle2 * 35) + T_mantle2
        T2_Pmax = (P_max * grad_mantle2 * 35) + T_mantle2

    # Determine the grid dimensions
    rows = len(unique_P)
    cols = len(unique_T)

    # Create an empty grid to store the reshaped parameter_values
    grid = np.empty((rows, cols))

    # Reshape the parameter_values to match the grid dimensions
    for i, p in enumerate(unique_P):
        for j, t in enumerate(unique_T):
            if (
                   ((t <= T1_Pmax) and (p >= geotherm1[j])) or
                   ((t >= T2_Pmin) and (p <= geotherm2[j]))
            ):
                index = None
            else:
                index = np.where((P == p) & (T == t))[0]

            if index is not None:
                grid[i, j] = parameter_values[index[0]]
            else:
                grid[i, j] = np.nan

    return grid

# Plot MAGEMin PT Diagram
def visualize_2d_pt_grid(
        P,
        T,
        grid,
        parameter,
        geotherm=False,
        geotherm_linetype="-.",
        geotherm_color="white",
        T_unit="K",
        P_unit="GPa",
        title=None,
        palette="blues",
        color_discrete=False,
        color_reverse=False,
        vmin=None,
        vmax=None,
        figwidth=6.3,
        figheight=4.725,
        fontsize=22,
        filename=None,
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Plot the results of a pseudosection calculation.

    Args:
        P (list or array-like): A 1D array or list of P values.
        T (list or array-like): A 1D array or list of T values.
        grid (numpy.ndarray): A 2D array representing the pseudosection grid.
        parameter (str): The parameter to plot. If "StableSolutions", phase assemblages
            will be plotted, otherwise, the specified parameter values will be plotted.
        title (str, optional): The title of the plot.
        palette (str, optional): The color palette to use for the plot.
        color_discrete (bool, optional): Whether to use a discrete color palette.
        color_reverse (bool, optional): Whether to reverse the color palette.
        filename (str, optional): If provided, the plot will be saved to the specified file.
        fig_dir (str, optional): The directory to save the plot (default is "./figs").

    Returns:
        None
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    if color_discrete:
        # Discrete color palette
        num_colors = len(np.unique(grid))

        if palette == "viridis":
            if color_reverse:
                pal = plt.cm.get_cmap("viridis_r", num_colors)
            else:
                pal = plt.cm.get_cmap("viridis", num_colors)
        elif palette == "bone":
            if color_reverse:
                pal = plt.cm.get_cmap("bone_r", num_colors)
            else:
                pal = plt.cm.get_cmap("bone", num_colors)
        elif palette == "pink":
            if color_reverse:
                pal = plt.cm.get_cmap("pink_r", num_colors)
            else:
                pal = plt.cm.get_cmap("pink", num_colors)
        elif palette == "seismic":
            if color_reverse:
                pal = plt.cm.get_cmap("seismic_r", num_colors)
            else:
                pal = plt.cm.get_cmap("seismic", num_colors)
        elif palette == "grey":
            if color_reverse:
                pal = plt.cm.get_cmap("Greys_r", num_colors)
            else:
                pal = plt.cm.get_cmap("Greys", num_colors)
        elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
            if color_reverse:
                pal = plt.cm.get_cmap("Blues_r", num_colors)
            else:
                pal = plt.cm.get_cmap("Blues", num_colors)

        # Descritize
        color_palette = pal(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(color_palette)

        # Set nan color
        cmap.set_bad(color="white")

        # Plot as a raster using imshow
        fig, ax = plt.subplots(figsize=(figwidth, figheight))
        im = ax.imshow(
            grid,
            extent=[
                np.array(T).min(),
                np.array(T).max(),
                np.array(P).min(),
                np.array(P).max()
            ],
            aspect="auto",
            cmap=cmap,
            origin="lower",
            vmin=1,
            vmax=num_colors + 1
        )
        ax.set_xlabel(f"T ({T_unit})")
        ax.set_ylabel(f"P ({P_unit})")
        plt.colorbar(
            im,
            ax=ax,
            ticks=np.arange(1, num_colors + 1, num_colors // 4),
            label=""
        )
    else:
        # Continuous color palette
        if palette == "viridis":
            if color_reverse:
                cmap = "viridis_r"
            else:
                cmap = "viridis"
        elif palette == "bone":
            if color_reverse:
                cmap = "bone_r"
            else:
                cmap = "bone"
        elif palette == "pink":
            if color_reverse:
                cmap = "pink_r"
            else:
                cmap = "pink"
        elif palette == "seismic":
            if color_reverse:
                cmap = "seismic_r"
            else:
                cmap = "seismic"
        elif palette == "grey":
            if color_reverse:
                cmap = "Greys_r"
            else:
                cmap = "Greys"
        elif palette not in ["viridis", "grey", "bone", "pink", "seismic"]:
            if color_reverse:
                cmap="Blues_r"
            else:
                cmap="Blues"

        # Adjust diverging colorscale to center on zero
        if palette == "seismic":
            vmin=-np.max(np.abs(grid[np.logical_not(np.isnan(grid))]))
            vmax=np.max(np.abs(grid[np.logical_not(np.isnan(grid))]))
        else:
            vmin=vmin
            vmax=vmax

        # Set nan color
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color="white")

        # Plot as a raster using imshow
        fig, ax = plt.subplots()
        im = ax.imshow(
            grid,
            extent=[
                np.array(T).min(),
                np.array(T).max(),
                np.array(P).min(),
                np.array(P).max()
            ],
            aspect="auto",
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_xlabel(f"T ({T_unit})")
        ax.set_ylabel(f"P ({P_unit})")
        if palette == "seismic":
            cbar = plt.colorbar(
                im,
                ax=ax,
                ticks=[vmin, 0, vmax],
                label=""
            )
        else:
            cbar = plt.colorbar(
                im,
                ax=ax,
                ticks=np.linspace(vmin, vmax, num=4),
                label=""
            )
        if parameter in ["Vp", "Vs", "DensityOfFullAssemblage"]:
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        if palette == "seismic":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

    # Add title
    if title:
        plt.title(title)

    # Add geotherms
    if geotherm:

        # Calculate geotherms
        P_arrays, T_arrays = crop_geotherms(P, T)

        # Crop each geotherm array based on PT bounds
        for i in range(len(P_arrays)):
            # Add geotherm to plot
            if i == len(P_arrays) - 1:
                plt.plot(
                    T_arrays[i],
                    P_arrays[i],
                    geotherm_linetype,
                    color=geotherm_color,
                    linewidth=2
                )
            else:
                plt.plot(
                    T_arrays[i],
                    P_arrays[i],
                    geotherm_linetype,
                    color=geotherm_color,
                    linewidth=2
                )

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

# Plot Harker diagram with density contours using seaborn
def visualize_earthchem_data(
        datafile,
        x_oxide="SiO2",
        y_oxide=["MgO", "FeO", "CaO", "Al2O3"],
        fontsize=12,
        filename="earthchem-samples-harker-diagram.png",
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Plot Harker diagrams with density contours using seaborn.

    Parameters:
        datafile (str): Path to the geochemical datafile in .csv format.
        x_oxide (str): The x-axis oxide for the Harker diagram.
        y_oxide (str or list): The y-axis oxide(s) for the Harker diagram.
            Can be a single oxide or a list of oxides.
        filename (str, optional): The filename to save the plot. If not provided,
            the plot will be displayed interactively.
        fig_dir (str, optional): The directory to save the plot. Default is "./figs".

    Returns:
        None

    Notes:
        - The function creates a directory named "figs" to save the plots.
        - The function uses seaborn "dark" style, palette, and "talk" context for the plots.
        - The function creates a grid of subplots based on the number of y-oxides specified.
        - Density contours and scatter plots (Harker diagrams) are added to each subplot.
        - Legends are shown only for the first subplot.
        - The plot can be saved to a file if a filename is provided.
    """
    # Ignore warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # Read geochemical data
    data = read_geochemical_data(datafile)
    data = data.rename(columns={"COMPOSITION": "Rock Type"})

    # Create figs dir
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Create a grid of subplots
    num_plots = len(y_oxide)
    if num_plots == 1:
        num_cols = 1
    elif ((num_plots > 1) and (num_plots <= 4)):
        num_cols = 2
    elif ((num_plots > 4) and (num_plots <= 9)):
        num_cols = 3
    elif ((num_plots > 9) and (num_plots <= 16)):
        num_cols = 4
    else:
        num_cols = 5
    num_rows = (num_plots + 1) // num_cols

    # Total figure size
    fig_width = 3.8 * num_cols
    fig_height = 3.15 * num_rows

    # Draw plots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    for i, y in enumerate(y_oxide):
        ax = axes[i]

        # Add density contours
        sns.kdeplot(
            data=data,
            x=x_oxide.upper(),
            y=y.upper(),
            hue="Rock Type",
            hue_order=["ultramafic", "mafic"],
            fill=True,
            ax=ax
        )

        # X-Y scatter (Harker diagram)
        sns.scatterplot(
            data=data,
            x=x_oxide.upper(),
            y=y.upper(),
            hue="Rock Type",
            hue_order=["ultramafic", "mafic"],
            linewidth=0,
            s=5,
            legend=False,
            ax=ax
        )

        ax.set_xlabel(f"{x_oxide} (wt%)")
        ax.set_ylabel(f"{y} (wt%)")
        ax.get_legend().set_title("")

        # Show legend only for the first subplot
        if i > 0:
            ax.get_legend().remove()

    # Remove empty subplots
    if num_plots < len(axes):
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

# Visualize benchmark comp times
def visualize_benchmark_gfem_times(
        datafile,
        palette="tab10",
        fontsize=12,
        figwidth=6.3,
        figheight=3.54,
        filename="benchmark-gfem-efficiency.png",
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Visualize benchmark computation times.

    Parameters:
        datafile (str): Path to the benchmark data file in .csv format.
        palette (str, optional): The color palette for sample colors. Default is "tab10".
        fontsize (int, optional): The font size for text elements in the plot. Default is 12.
        filename (str, optional): The filename to save the plot. If not provided,
            the plot will be displayed interactively. Default is "efficiency.png".
        fig_dir (str, optional): The directory to save the plot. Default is "./figs".

    Returns:
        None

    Notes:
        - The function creates a directory named "figs" to save the plots.
        - The function reads the data from the provided datafile in .csv format.
        - The function sets plot styles and settings.
        - The function creates a dictionary to map samples to colors using a colormap.
        - The function groups the data by sample.
        - The function filters out rows with missing time values in both mgm and ppx columns.
        - The function extracts x and y values from the filtered data for both mgm and ppx columns.
        - The function plots the data points and connects them with lines.
        - The function sets labels, title, and x-axis tick values.
        - The plot can be saved to a file if a filename is provided.
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read data
    data = pd.read_csv(datafile)

    # Arrange data by training dataset resolution and sample
    data.sort_values(by=["grid", "sample", "program"], inplace=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Create a dictionary to map samples to colors using a colormap
    colormap = plt.cm.get_cmap(palette)
    sample_colors = {
        "DMM": colormap(0),
        "NMORB": colormap(1),
        "PUM": colormap(2),
        "RE46": colormap(3)
    }

    # Group the data by sample
    grouped_data = data.groupby(["sample"])

    # Create a list to store the legend labels and handles
    legend_labels = []
    legend_handles = []

    # Plot the data for MAGEMin lines
    for group, group_data in grouped_data:
        sample_val = group[0]
        color_val = sample_colors[sample_val]

        # Filter out rows with missing time values for mgm column
        mgm_data = group_data[group_data["program"] == "magemin"]
        mgm_x = mgm_data["grid"]
        mgm_y = mgm_data["time"]

        # Filter out rows with missing time values for ppx column
        ppx_data = group_data[group_data["program"] == "perplex"]
        ppx_x = ppx_data["grid"]
        ppx_y = ppx_data["time"]

        # Plot mgm data points and connect them with lines
        line_mgm, = plt.plot(
            mgm_x, mgm_y,
            marker="o",
            color=color_val,
            linestyle="-",
            label=f"[MAGEMin] {sample_val}"
        )
        legend_handles.append(line_mgm)
        legend_labels.append(f"[MAGEMin] {sample_val}")

        # Plot ppx data points and connect them with lines
        line_ppx, = plt.plot(
            ppx_x, ppx_y,
            marker="s",
            color=color_val,
            linestyle="--",
            label=f"[Perple_X] {sample_val}"
        )
        legend_handles.append(line_ppx)
        legend_labels.append(f"[Perple_X] {sample_val}")

    # Set labels and title
    plt.xlabel("Number of Minimizations (PT Grid Size)")
    plt.ylabel("Elapsed Time (s)")
    plt.title("GFEM Efficiency")
    plt.xscale("log")
    plt.yscale("log")

    # Create the legend with the desired order
    plt.legend(
        legend_handles,
        legend_labels,
        title="",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    # Adjust the figure size
    fig = plt.gcf()
    fig.set_size_inches(figwidth, figheight)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

# Visualize training data PT range and Clapeyron slopes
def visualize_training_PT_range(
        P_unit="GPa",
        T_unit="K",
        palette="tab10",
        fontsize=12,
        figwidth=6.3,
        figheight=3.54,
        filename="training-dataset-design.png",
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Generate a plot to visualize the training data PT range and Clapeyron slopes.

    Parameters:
        - palette (str): Color palette for the plot. Default is "tab10".
        - fontsize (int): Font size for the plot. Default is 14.
        - figwidth (float): Width of the figure in inches. Default is 6.3.
        - figheight (float): Height of the figure in inches. Default is 3.54.
        - filename (str): Name of the file to save the plot.
        - fig_dir (str): Directory path to save the plot file.

    Returns:
        None. The plot is either saved to a file or displayed on the screen.
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # T range
    T = np.arange(0, 3001)
    T_min, T_max = 773, 2273

    # Define P range
    P_min, P_max = 1.0, 28.0

    # Olivine --> Ringwoodite Clapeyron slopes
    references_410 = {
        "[410] Akaogi89": [0.001, 0.002],
        "[410] Katsura89": [0.0025],
        "[410] Morishima94": [0.0034, 0.0038]
    }

    # Ringwoodite --> Bridgmanite + Ferropericlase Clapeyron slopes
    references_660 = {
        "[660] Ito82": [-0.002],
        "[660] Ito89 & Hirose02": [-0.0028],
        "[660] Ito90": [-0.002, -0.006],
        "[660] Katsura03": [-0.0004, -0.002],
        "[660] Akaogi07": [-0.0024, -0.0028]
    }

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Legend colors
    colormap = plt.cm.get_cmap(palette)
    colors = [
        colormap(0),
        colormap(1),
        colormap(2),
        colormap(3),
        colormap(4),
        colormap(5),
        colormap(6),
        colormap(7),
        colormap(8)
    ]

    # Calculate phase boundaries:
    # Olivine --> Ringwoodite
    lines_410 = []
    labels_410 = set()
    for i, (ref, c_values) in enumerate(references_410.items()):
        ref_lines = []
        for j, c in enumerate(c_values):
            P = (T - 1758) * c + 13.4
            ref_lines.append(P)
            label = f"{ref}"
            labels_410.add(label)
        lines_410.append(ref_lines)

    # Ringwoodite --> Bridgmanite + Ferropericlase
    lines_660 = []
    labels_660 = set()
    for i, (ref, c_values) in enumerate(references_660.items()):
        ref_lines = []
        for j, c in enumerate(c_values):
            P = (T - 1883) * c + 23.0
            ref_lines.append(P)
            label = f"{ref}"
            labels_660.add(label)
        lines_660.append(ref_lines)

    # Plotting
    plt.figure()

    # Map labels to colors
    label_color_mapping = {}

    # Olivine --> Ringwoodite
    for i, (ref, ref_lines) in enumerate(zip(references_410.keys(), lines_410)):
        color = colors[i % len(colors)]
        for j, line in enumerate(ref_lines):
            label = f"{ref}" if j == 0 else None
            plt.plot(
                T[(T >= 1200) & (T <= 2000)],
                line[(T >= 1200) & (T <= 2000)],
                color=color,
                label=label
            )
            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Ringwoodite --> Bridgmanite + Ferropericlase
    for j, (ref, ref_lines) in enumerate(zip(references_660.keys(), lines_660)):
        color = colors[j + i + 1 % len(colors)]
        for j, line in enumerate(ref_lines):
            label = f"{ref}" if j == 0 else None
            plt.plot(
                T[(T >= 1200) & (T <= 2000)],
                line[(T >= 1200) & (T <= 2000)],
                color=color,
                label=label
            )
            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Plot shaded rectangle for PT range of training dataset
    fill = plt.fill_between(
        T,
        P_min,
        P_max,
        where=(T >= T_min) & (T <= T_max),
        color="gray",
        alpha=0.2
    )

    # Mantle potential temps
    T_mantle1 = 0 + 273
    T_mantle2 = 1500 + 273

    # Thermal gradients
    grad_mantle1 = 1
    grad_mantle2 = 0.5

    # Calculate mantle geotherms
    geotherm1 = (T - T_mantle1) / (grad_mantle1 * 35)
    geotherm2 = (T - T_mantle2) / (grad_mantle2 * 35)

    # Find boundaries
    T1_Pmax = (P_max * grad_mantle1 * 35) + T_mantle1
    P1_Tmin = (T_min - T_mantle1) / (grad_mantle1 * 35)
    T2_Pmin = (P_min * grad_mantle2 * 35) + T_mantle2
    T2_Pmax = (P_max * grad_mantle2 * 35) + T_mantle2

    # Crop geotherms
    geotherm1_cropped = geotherm1[geotherm1 >= P1_Tmin]
    geotherm1_cropped = geotherm1_cropped[geotherm1_cropped <= P_max]
    geotherm2_cropped = geotherm2[geotherm2 >= P_min]
    geotherm2_cropped = geotherm2_cropped[geotherm2_cropped <= P_max]

    # Crop T vectors
    T_cropped_geotherm1= T[T >= T_min]
    T_cropped_geotherm1 = T_cropped_geotherm1[T_cropped_geotherm1 <= T1_Pmax]
    T_cropped_geotherm2= T[T >= T2_Pmin]
    T_cropped_geotherm2 = T_cropped_geotherm2[T_cropped_geotherm2 <= T2_Pmax]

    # Plot mantle geotherms
    plt.plot(T_cropped_geotherm1, geotherm1_cropped, "-", color="black")
    plt.plot(T_cropped_geotherm2, geotherm2_cropped, "--", color="black")

    # Interpolate the geotherms to have the same length as temperature vectors
    geotherm1_interp = np.interp(T_cropped_geotherm1, T, geotherm1)
    geotherm2_interp = np.interp(T_cropped_geotherm2, T, geotherm2)

    # Define the vertices for the polygon
    vertices = np.vstack(
        (
            np.vstack((T_cropped_geotherm1, geotherm1_interp)).T,
            (T_cropped_geotherm2[-1], geotherm2_interp[-1]),
            np.vstack((T_cropped_geotherm2[::-1], geotherm2_interp[::-1])).T,
            np.array([T_min, P_min]),
            (T_cropped_geotherm1[0], geotherm1_interp[0])
        )
    )

    # Fill the area within the polygon
    plt.fill(
        vertices[:, 0],
        vertices[:, 1],
        facecolor="blue",
        edgecolor="black",
        alpha=0.1
    )

    # Geotherm legend handles
    geotherm1_handle = mlines.Line2D(
        [], [], linestyle="-", color="black", label="Geotherm 1"
    )
    geotherm2_handle = mlines.Line2D(
        [], [], linestyle="--", color="black", label="Geotherm 2"
    )

    # Phase boundaries legend handles
    ref_line_handles = [
        mlines.Line2D([], [], color=color, label=label) \
        for label, color in label_color_mapping.items() if label
    ]

    # Add geotherms to legend handles
    ref_line_handles.extend([geotherm1_handle, geotherm2_handle])

    db_data_handle = mpatches.Patch(color="gray", alpha=0.2, label="Training Data Range")

    labels_660.add("Training Data Range")
    label_color_mapping["Training Data Range"] = "gray"

    training_data_handle = mpatches.Patch(
        facecolor="blue",
        edgecolor="black",
        alpha=0.1,
        label="Mantle Conditions"
    )

    labels_660.add("Mantle Conditions")
    label_color_mapping["Mantle Conditions"] = "gray"

    # Define the desired order of the legend items
    desired_order = [
        "Training Data Range",
        "Mantle Conditions",
        "[410] Akaogi89",
        "[410] Katsura89",
        "[410] Morishima94",
        "[660] Ito82",
        "[660] Ito89 & Hirose02",
        "[660] Ito90",
        "[660] Katsura03",
        "[660] Akaogi07",
        "Geotherm 1",
        "Geotherm 2"
    ]

    # Sort the legend handles based on the desired order
    legend_handles = sorted(
        ref_line_handles + [db_data_handle, training_data_handle],
        key=lambda x: desired_order.index(x.get_label())
    )

    plt.xlabel(f"Temperature ({T_unit})")
    plt.ylabel(f"Pressure ({P_unit})")
    plt.title("RocML Traning Dataset")
    plt.xlim(700, 2346)
    plt.ylim(0, 29)

    # Move the legend outside the plot to the right
    plt.legend(
        title="",
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    # Adjust the figure size
    fig = plt.gcf()
    fig.set_size_inches(figwidth, figheight)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

# Visualize MGM and PPX comparisons with PREM model
def visualize_PREM(
        datafile,
        parameter,
        param_unit,
        results_mgm=None,
        results_ppx=None,
        results_rocml=None,
        model=None,
        geotherm_threshold=0.1,
        P_unit="GPa",
        depth=True,
        metrics=None,
        palette="tab10",
        title=None,
        figwidth=6.3,
        figheight=4.725,
        fontsize=22,
        filename=None,
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Visualize MGM and PPX comparisons with PREM model.

    This function creates a plot comparing the parameter specified in results_mgm and
    results_ppx with the PREM (Preliminary Reference Earth Model) data.
    The function reads data from the provided CSV file, applies transformations to the data
    and units, and subsets the data based on a geotherm threshold.

    Parameters:
        datafile (str): The file path to the CSV data containing the PREM model.
        results_mgm (dict): A dictionary containing the results of the MGM model with
        pressure (P) and temperature (T) values and the parameter of interest.
        results_ppx (dict): A dictionary containing the results of the PPX model with
        pressure (P) and temperature (T) values and the parameter of interest.
        parameter (str): The parameter of interest to be plotted (e.g.,
        DensityOfFullAssemblage).
        param_unit (str): The unit of the parameter to be displayed in the plot
        (e.g., "kg/m^3").
        geotherm_threshold (float, optional): The geotherm threshold used to subset the data
        along the geotherm. Default is 1.
        P_unit (str, optional): The unit of pressure to be displayed in the plot.
        Default is "GPa".
        palette (str, optional): The name of the color palette for plotting.
        Default is "tab10".
        title (str, optional): The title of the plot. Default is None.
        figwidth (float, optional): The width of the plot figure in inches. Default is 6.3.
        figheight (float, optional): The height of the plot figure in inches.
        Default is 4.725.
        fontsize (int, optional): The font size used in the plot. Default is 22.
        filename (str, optional): The name of the file to save the plot. If None, the plot
        will be displayed instead of saving. Default is None.
        fig_dir (str, optional): The directory to save the plot file. Default is
        "<current working directory>/figs".

    Returns:
        None: The function generates a plot comparing the MGM and PPX models with the PREM
        data.

    Note:
        - The function relies on the matplotlib and pandas libraries for data visualization
        and manipulation.
        - Make sure the datafile contains the necessary columns (e.g., "depth", "P") for
        calculating pressure and depth conversions.
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read the CSV file into a pandas DataFrame
    df_prem = pd.read_csv(datafile)

    # Extract depth and parameter values
    param_prem = df_prem[parameter]
    depth_prem = df_prem["depth"]

    # Transform depth to pressure
    P_prem = depth_prem / 30

    # Check valid lists and get minimum and maximum values
    P_results = [
        results["P"] for results in
        [results_mgm, results_ppx, results_rocml] if
        results is not None
    ]

    if not P_results:
        raise ValueError(
            "No valid results lists for MAGEMin, Perple_X, or ML model ...\n"
            "Please provide at least one list of results"
        )

    P_min = min(min(lst) for lst in P_results) / 10
    P_max = max(max(lst) for lst in P_results) / 10

    param_results = [
        results[parameter] for results in
        [results_mgm, results_ppx, results_rocml] if
        results is not None
    ]

    param_min = min(min(lst) for lst in param_results)
    param_max = max(max(lst) for lst in param_results)

    # Crop pressure and parameter values at min max lims of results
    mask = P_prem >= P_min
    P_prem = P_prem[mask]
    param_prem = param_prem[mask]

    mask = P_prem <= P_max + 1
    P_prem = P_prem[mask]
    param_prem = param_prem[mask]

    # Extract parameter values along a geotherm
    if results_mgm:
        P_mgm, T_mgm, param_mgm = extract_info_geotherm(
            results_mgm, parameter, threshold=geotherm_threshold
        )

    if results_ppx:
        P_ppx, T_ppx, param_ppx = extract_info_geotherm(
            results_ppx, parameter, threshold=geotherm_threshold
        )

    if results_rocml:
        P_rocml, T_rocml, param_rocml = extract_info_geotherm(
            results_rocml, parameter, threshold=geotherm_threshold
        )

    # Transform units
    if parameter == "DensityOfFullAssemblage":
        param_min = param_min / 1000
        param_max = param_max / 1000

        if results_mgm:
            param_mgm = param_mgm / 1000

        if results_ppx:
            param_ppx = param_ppx / 1000

        if results_rocml:
            param_rocml = param_rocml / 1000

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Colormap
    colormap = plt.cm.get_cmap(palette)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

    # Plot PREM data on the primary y-axis
    ax1.plot(param_prem, P_prem, "-", linewidth=3, color="black", label="PREM")

    if results_mgm:
        ax1.plot(param_mgm, P_mgm, "-", linewidth=3, color=colormap(0), label="MAGEMin")
    if results_ppx:
        ax1.plot(param_ppx, P_ppx, "-", linewidth=3, color=colormap(2), label="Perple_X")
    if results_rocml:
        ax1.plot(param_rocml, P_rocml, "-", linewidth=3, color=colormap(1), label=f"{model}")

    if parameter == "DensityOfFullAssemblage":
        parameter_label = "Density"
    else:
        parameter_label = parameter

    ax1.set_xlabel(f"{parameter_label } ({param_unit})")
    ax1.set_ylabel(f"P ({P_unit})")
    ax1.set_xlim(param_min - (param_min * 0.05), param_max + (param_max * 0.05))
    ax1.set_xticks(np.linspace(param_min, param_max, num=4))
    if parameter in ["Vp", "Vs", "DensityOfFullAssemblage"]:
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    if metrics is not None:
        # Vertical text spacing
        text_margin_x = 0.04
        text_margin_y = 0.15
        text_spacing_y = 0.1

        # Get metrics
        rmse_mean, r2_mean = metrics

        # Add R-squared and RMSE values as text annotations in the plot
        plt.text(
            1 - text_margin_x,
            text_margin_y - (text_spacing_y * 0),
            f"R$^2$: {r2_mean:.3f}",
            transform=plt.gca().transAxes,
            fontsize=fontsize * 0.833,
            horizontalalignment="right",
            verticalalignment="bottom"
        )
        plt.text(
            1 - text_margin_x,
            text_margin_y - (text_spacing_y * 1),
            f"RMSE: {rmse_mean:.3f}",
            transform=plt.gca().transAxes,
            fontsize=fontsize * 0.833,
            horizontalalignment="right",
            verticalalignment="bottom"
        )

    if depth:
        # Convert the primary y-axis data (pressure) to depth
        depth_conversion = lambda P: P * 30
        depth_values = depth_conversion(P_prem)

        # Create the secondary y-axis and plot depth on it
        ax2 = ax1.secondary_yaxis("right", functions=(depth_conversion, depth_conversion))
        ax2.set_yticks([410, 660])
        ax2.set_ylabel("Depth (km)")

    plt.legend()
    if title:
        plt.title(title)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}")
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

# Plot GFEM results
def visualize_GFEM(
        program,
        sample_id,
        res,
        dataset,
        parameters,
        mask_geotherm=False,
        palette="bone",
        out_dir=f"{os.getcwd()}/runs",
        fig_dir=f"{os.getcwd()}/figs",
        data_dir=f"{os.getcwd()}/assets/data"):
    """
    Visualize GFEM (Gibbs Free Energy Minimization) results for a given program and sample.

    Parameters:
        program (str): The GFEM program used for the analysis. Supported values are
                       "MAGEMin" and "Perple_X".
        sample_id (str): The identifier of the sample for which GFEM results are to be
                         visualized.
        parameters (list): A list of parameters to be visualized. Supported parameters
                           depend on the GFEM program used.
        palette (str, optional): The color palette for the plots. Default is "bone".
        out_dir (str, optional): Directory to store intermediate output files from the GFEM
                                 program. Default is "./runs".
        fig_dir (str, optional): Directory to save the generated visualization figures.
                                 Default is "./figs".
        data_dir (str, optional): Directory containing data files needed for visualization.
                                  Default is "./assets/data".

    Raises:
        ValueError: If an invalid value for 'program' is provided. It must be either
                    "MAGEMin" or "Perple_X".

    Notes:
        - This function first processes the GFEM results based on the selected 'program'.
        - It then transforms the results to appropriate units and creates 2D numpy arrays
          for visualization.
        - The visualization is done using the 'visualize_2d_pt_grid' function, which generates
          plots of the parameters on a P-T (pressure-temperature) grid for the given sample.
    """
    # Get GFEM results
    if program == "MAGEMin":
        # Get MAGEMin results
        results = process_magemin_grid(sample_id, dataset, res, out_dir)
    elif program == "Perple_X":
        # Get perplex results
        results = process_perplex_grid(
            f"{out_dir}/{sample_id}/perplex_{dataset}_{res}/{sample_id}_grid.tab",
            f"{out_dir}/{sample_id}/perplex_{dataset}_{res}/{sample_id}_assemblages.txt"
        )
    else:
        raise ValueError(
            "Invalid program argument ...\n"
            "program must be MAGEMin or Perple_X"
        )

    # Get PT values MAGEMin and transform units
    P = [P / 10 for P in results["P"]]
    T = [T + 273 for T in results["T"]]

    for parameter in parameters:
        # Transform results into 2D numpy arrays
        if parameter == "StableSolutions":
            # Encode unique phase assemblages MAGEMin
            encoded, unique = encode_phases(
                results[parameter],
                filename=f"{data_dir}/{sample_id}_{program}_assemblages.csv"
            )
            grid = create_PT_grid(P, T, encoded, mask_geotherm)
        else:
            grid = create_PT_grid(P, T, results[parameter], mask_geotherm)

        # Transform units
        if parameter == "DensityOfFullAssemblage":
            grid = grid / 1000

        # Use discrete colorscale
        if parameter in ["StableSolutions", "StableVariance"]:
            color_discrete = True
        else:
            color_discrete = False

        # Reverse color scale
        if palette in ["grey"]:
            if parameter in ["StableVariance"]:
                color_reverse = True
            else:
                color_reverse = False
        else:
            if parameter in ["StableVariance"]:
                color_reverse = False
            else:
                color_reverse = True

        # Set colorbar limits for better comparisons
        if not color_discrete:
            vmin=np.min(grid[np.logical_not(np.isnan(grid))])
            vmax=np.max(grid[np.logical_not(np.isnan(grid))])
        else:
            num_colors = len(np.unique(grid))
            vmin = 1
            vmax = num_colors + 1

        # Plot PT grid MAGEMin
        visualize_2d_pt_grid(
            P,
            T,
            grid,
            parameter,
            title=program,
            palette=palette,
            color_discrete=color_discrete,
            color_reverse=color_reverse,
            vmin=vmin,
            vmax=vmax,
            filename=f"{program}-{sample_id}-{dataset}-{parameter}.png",
            fig_dir=fig_dir
        )

# Plot GFEM results diff
def visualize_GFEM_diff(
        sample_id,
        res,
        dataset,
        parameters,
        mask_geotherm=False,
        palette="bone",
        out_dir=f"{os.getcwd()}/runs",
        fig_dir=f"{os.getcwd()}/figs",
        data_dir=f"{os.getcwd()}/assets/data"):
    """
    Visualize the difference between GFEM results from two programs for a given sample.

    Parameters:
        sample_id (str): The identifier of the sample for which GFEM results are to be
                         compared.
        parameters (list): A list of parameters to be compared. Supported parameters depend
                           on the GFEM programs used.
        palette (str, optional): The color palette for the plots. Default is "bone".
        out_dir (str, optional): Directory to store intermediate output files from the GFEM
                                 programs. Default is "./runs".
        fig_dir (str, optional): Directory to save the generated visualization figures.
                                 Default is "./figs".
        data_dir (str, optional): Directory containing data files needed for visualization.
                                  Default is "./assets/data".

    Notes:
        - This function retrieves GFEM results from two programs, "MAGEMin" and "Perple_X"
          for the given sample.
        - It transforms the results to appropriate units and creates 2D numpy arrays for
          visualization.
        - The visualization is done using the 'visualize_2d_pt_grid' function, which generates
          plots of the differences
          between the GFEM results on a P-T (pressure-temperature) grid.
        - The function also computes and visualizes the normalized difference and the
          maximum gradient between the results of the two programs for a more comprehensive
          comparison.
    """
    # Get MAGEMin results
    results_mgm = process_magemin_grid(sample_id, dataset, res, out_dir)

    # Get perplex results
    results_ppx = process_perplex_grid(
        f"{out_dir}/{sample_id}/perplex_{dataset}_{res}/{sample_id}_grid.tab",
        f"{out_dir}/{sample_id}/perplex_{dataset}_{res}/{sample_id}_assemblages.txt"
    )

    # Get PT values MAGEMin and transform units
    P_mgm = [P / 10 for P in results_mgm["P"]]
    T_mgm = [T + 273 for T in results_mgm["T"]]

    # Get PT values perplex and transform units
    P_ppx = [P / 10 for P in results_ppx["P"]]
    T_ppx = [T + 273 for T in results_ppx["T"]]

    for parameter in parameters:
        # Transform results into 2D numpy arrays
        if parameter == "StableSolutions":
            # Encode unique phase assemblages MAGEMin
            encoded_mgm, unique_mgm = encode_phases(results_mgm[parameter])
            grid_mgm = create_PT_grid(P_mgm, T_mgm, encoded_mgm, mask_geotherm)

            # Encode unique phase assemblages perplex
            encoded_ppx, unique_ppx = encode_phases(results_ppx[parameter])
            grid_ppx = create_PT_grid(P_ppx, T_ppx, encoded_ppx, mask_geotherm)
        else:
            grid_mgm = create_PT_grid(P_mgm, T_mgm, results_mgm[parameter], mask_geotherm)
            grid_ppx = create_PT_grid(P_ppx, T_ppx, results_ppx[parameter], mask_geotherm)

        # Transform units
        if parameter == "DensityOfFullAssemblage":
            grid_mgm = grid_mgm / 1000
            grid_ppx = grid_ppx / 1000

        # Use discrete colorscale
        if parameter in ["StableSolutions", "StableVariance"]:
            color_discrete = True
        else:
            color_discrete = False

        # Reverse color scale
        if palette in ["grey"]:
            if parameter in ["StableVariance"]:
                color_reverse = True
            else:
                color_reverse = False
        else:
            if parameter in ["StableVariance"]:
                color_reverse = False
            else:
                color_reverse = True

        # Set colorbar limits for better comparisons
        if not color_discrete:
            vmin_mgm=np.min(grid_mgm[np.logical_not(np.isnan(grid_mgm))])
            vmax_mgm=np.max(grid_mgm[np.logical_not(np.isnan(grid_mgm))])
            vmin_ppx=np.min(grid_ppx[np.logical_not(np.isnan(grid_ppx))])
            vmax_ppx=np.max(grid_ppx[np.logical_not(np.isnan(grid_ppx))])

            vmin = min(vmin_mgm, vmin_ppx)
            vmax = max(vmax_mgm, vmax_ppx)
        else:
            num_colors_mgm = len(np.unique(grid_mgm))
            num_colors_ppx = len(np.unique(grid_ppx))
            vmin = 1
            vmax = max(num_colors_mgm, num_colors_ppx) + 1

        if not color_discrete:
            # Define a filter to ignore the specific warning
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")

            # Compute normalized diff
            mask = ~np.isnan(grid_mgm) & ~np.isnan(grid_ppx)
            diff_norm = (grid_mgm - grid_ppx) / ((grid_mgm + grid_ppx) / 2) * 100
            diff_norm[~mask] = np.nan

            # Plot PT grid normalized diff mgm-ppx
            visualize_2d_pt_grid(
                P_ppx,
                T_ppx,
                diff_norm,
                parameter,
                title="Percent Difference",
                palette="seismic",
                color_discrete=color_discrete,
                color_reverse=False,
                vmin=vmin,
                vmax=vmax,
                filename=f"diff-{sample_id}-{dataset}-{parameter}.png",
                fig_dir=fig_dir
            )

            # Set geotherm threshold for extracting depth profiles
            if res <= 8:
                geotherm_threshold = 4
            elif res <= 16:
                geotherm_threshold = 2
            elif res <= 32:
                geotherm_threshold = 1
            elif res <= 64:
                geotherm_threshold = 0.5
            elif res <= 128:
                geotherm_threshold = 0.25
            else:
                geotherm_threshold = 0.125

            # Plot PREM comparisons
            if parameter == "DensityOfFullAssemblage":
                visualize_PREM(
                    f"{data_dir}/prem.csv",
                    parameter,
                    "g/cm$^3$",
                    results_mgm=results_mgm,
                    results_ppx=results_ppx,
                    geotherm_threshold=geotherm_threshold,
                    title="PREM Comparison",
                    filename=f"prem-{sample_id}-{dataset}-{parameter}.png",
                    fig_dir=fig_dir
                )

            if parameter in ["Vp", "Vs"]:
                visualize_PREM(
                    f"{data_dir}/prem.csv",
                    parameter,
                    "km/s",
                    results_mgm=results_mgm,
                    results_ppx=results_ppx,
                    geotherm_threshold=geotherm_threshold,
                    title="PREM Comparison",
                    filename=f"prem-{sample_id}-{dataset}-{parameter}.png",
                    fig_dir=fig_dir
                )

# Define a function that performs the processing for a single fold
mp.set_start_method("fork")

def process_fold(fold_data, verbose=False):
    """
    Process a single fold of k-fold cross-validation for ML model.

    Parameters:
        fold_data (tuple): A tuple containing the necessary data for the k-fold cross
                           validation process. It should include the following elements:

            - fold_idx (int): The index of the current fold.
            - (train_index, test_index) (tuple): A tuple containing the training and testing
                                                 indices for the current fold.
            - kfolds (int): Total number of folds in the cross-validation.
            - X_scaled (array-like): The scaled feature data (input variables).
            - y_scaled (array-like): The scaled target data (output variable).
            - X_valid_scaled (array-like): The scaled validation feature data for evaluation.
            - model (model object): The ML model to be trained and evaluated.
            - scaler_y (Scaler object): The scaler used to normalize the target variable.
            - scaler_X (Scaler object): The scaler used to normalize the feature variables.
            - target_array (array-like): The original (non-scaled) target array for the
                                         entire dataset, used for evaluation.
            - W (int): The width of the grid used for reshaping predictions for
                       visualization.

    Returns:
        rmse_test (float): Root Mean Squared Error (RMSE) of the ML model's predictions on
                           the test dataset (non-scaled).
        rmse_valid (float): Root Mean Squared Error (RMSE) of the ML model's predictions on
                            the validation dataset (non-scaled).
        r2_test (float): R-squared (coefficient of determination) of the ML model's
                         predictions on the test dataset.
        r2_valid (float): R-squared (coefficient of determination) of the ML model's
                          predictions on the validation dataset.

    Notes:
        This function performs the following steps for a single fold of k-fold cross
        validation for ML models:
            1. Unpacks the necessary data from the 'fold_data' tuple.
            2. Trains the ML model on the training data.
            3. Makes predictions on the test and validation sets.
            4. Performs inverse scaling to obtain predictions in their original units.
            5. Evaluates the model's performance using RMSE and R-squared metrics for both
               test and validation sets.
            6. Prints the fold index, number of samples in the test and train sets, RMSE, and
               R-squared scores.
    """
    # Unpack arguments
    (
        (train_index, test_index),
        X_scaled,
        y_scaled,
        X_scaled_valid,
        y_scaled_valid,
        model,
        scaler_X,
        scaler_y,
        scaler_X_valid,
        scaler_y_valid
    ) = fold_data

    # Split the data into training and testing sets
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    X_valid, y_valid = X_scaled_valid, y_scaled_valid

    # Train ML model
    training_start_time = time.time()
    model.fit(X_train, y_train)
    training_end_time = time.time()

    training_time = (training_end_time - training_start_time) * 1000

    # Make predictions on the test dataset
    y_pred_scaled = model.predict(X_test)
    y_pred_scaled_valid = model.predict(X_valid)

    # Test inference time on single random PT datapoint from the test dataset
    rand_PT_point = X_test[np.random.choice(X_test.shape[0], 1, replace=False)]

    inference_start_time = time.time()
    single_PT_pred = model.predict(rand_PT_point)
    inference_end_time = time.time()

    inference_time = (inference_end_time - inference_start_time) * 1000

    # Inverse transform predictions
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    y_pred_original_valid = scaler_y_valid.inverse_transform(y_pred_scaled_valid)

    # Inverse transform test dataset
    y_test_original = scaler_y.inverse_transform(y_test)
    y_valid_original = scaler_y_valid.inverse_transform(y_valid)

    # Get the ranges for each of the parameters
    target_ranges = np.max(y_valid_original, axis=0) - np.min(y_valid_original, axis=0)

    # Calculate performance metrics to evaluate the model
    rmse_test = (np.sqrt(
        mean_squared_error(
            y_test_original,
            y_pred_original,
            multioutput="raw_values"
        )
    ) / target_ranges) * 100
    rmse_valid = (np.sqrt(
        mean_squared_error(
            y_valid_original,
            y_pred_original_valid,
            multioutput="raw_values"
        )
    ) / target_ranges) * 100
    r2_test = r2_score(y_test_original, y_pred_original, multioutput="raw_values")
    r2_valid = r2_score(y_valid_original, y_pred_original_valid, multioutput="raw_values")

    return rmse_test, r2_test, rmse_valid, r2_valid, training_time, inference_time

# Cross-validate and train RocMLs
def cv_rocml(
        features_array,
        targets_array,
        features_array_valid,
        targets_array_valid,
        parameters,
        units,
        program,
        sample_id,
        model="DT",
        tune=False,
        seed=42,
        kfolds=10,
        parallel=True,
        nprocs=cpu_count()-2,
        vmin=None,
        vmax=None,
        palette="bone",
        figwidth=6.3,
        figheight=4.725,
        fontsize=22,
        filename=None,
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Runs ML model regression on the provided feature and target arrays.

    Parameters:
        features_array (ndarray): The feature array with shape (W, W, 2), containing
                                  pressure and temperature features.
        target_array (ndarray): The target array with shape (W, W), containing the
                                corresponding target values.
        parameter (str): The name of the parameter being predicted.
        units (str): The units of the predicted target parameter.
        parameter_units (str): The units of the original target parameter for display
                               purposes.
        seed (int, optional): The random seed for reproducibility. Default is 42.
        figwidth (float, optional): The width of the plot in inches. Default is 6.3.
        figheight (float, optional): The height of the plot in inches. Default is 4.725.
        fontsize (int, optional): The base fontsize for text in the plot. Default is 22.
        filename (str, optional): The filename to save the plot. Default is None.
        fig_dir (str, optional): The directory to save the plot. Default is "./figs".

    Returns:
        tuple: A tuple containing the trained ML model and evaluation metrics (rmse,
               r2_score).
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams['figure.constrained_layout.use'] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Colormap
    cmap = plt.cm.get_cmap("bone_r")
    cmap.set_bad(color="white")

    # Check parameters list for density
    try:
        index_rho = parameters.index("DensityOfFullAssemblage")
    except:
        print("Density not found in parameters")

    # Check parameters list for Vp
    try:
        index_vp = parameters.index("Vp")
    except:
        print("Vp not found in parameters")

    # Check parameters list for Vs
    try:
        index_vs = parameters.index("Vs")
    except:
        print("Vs not found in parameters")

    # Check parameters list for melt
    try:
        index_liq = parameters.index("LiquidFraction")
    except:
        print("LiquidFraction not found in parameters")

    # Transform units
    if index_rho is not None:
        targets_array[:,:,index_rho] = targets_array[:,:,index_rho] / 1000
        targets_array_valid[:,:,index_rho] = targets_array_valid[:,:,index_rho] / 1000

    if index_liq is not None:
        targets_array[:,:,index_liq] = targets_array[:,:,index_liq] * 100
        targets_array_valid[:,:,index_liq] = targets_array_valid[:,:,index_liq] * 100

    # Label units
    if units is None:
        units_labels = ["" for unit in units]
    else:
        units_labels = [f"({unit})" for unit in units]

    # Reshape the features_array and targets_array
    n_features = targets_array.shape[2]
    W = features_array.shape[0]
    X = features_array.reshape(W*W, 2)
    y = targets_array.reshape(W*W, n_features)
    X_valid = features_array_valid.reshape(W*W, 2)
    y_valid = targets_array_valid.reshape(W*W, n_features)

    # Remove nans
    mask = np.any([np.isnan(y[:,i]) for i in range(y.shape[1])], axis=0)
    X, y = X[~mask,:], y[~mask,:]

    n_nans_training = sum(mask)

    mask = np.any([np.isnan(y_valid[:,i]) for i in range(y_valid.shape[1])], axis=0)
    X_valid, y_valid = X_valid[~mask,:], y_valid[~mask,:]

    n_nans_valid = sum(mask)

    # Scale the feature array
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    scaler_X_valid = StandardScaler()
    scaler_y_valid = StandardScaler()

    # Scale feature and validation arrays
    X_scaled = scaler_X.fit_transform(X)
    X_scaled_valid = scaler_X_valid.fit_transform(X_valid)

    # Scale the target array
    y_scaled = scaler_y.fit_transform(y)
    y_scaled_valid = scaler_y_valid.fit_transform(y_valid)

    # Save model label string
    model_label = model

    if model_label == "KN":
        model_label_full = "K Nearest"
    elif model_label == "RF":
        model_label_full = "Random Forest"
    elif model_label == "DT":
        model_label_full = "Decision Tree"
    elif model_label == "NN1":
        model_label_full = "Neural Net 1L"
    elif model_label == "NN2":
        model_label_full = "Neural Net 2L"
    elif model_label == "NN3":
        model_label_full = "Neural Net 3L"

    # Define number of processors
    if not parallel:
        nprocs = 1
    elif parallel:
        if nprocs is None:
            nprocs = cpu_count() - 2
        else:
            nprocs = nprocs

    # Define ML models without tuning
    if not tune:
        if model_label == "KN":
            model = KNeighborsRegressor(n_neighbors=4, weights="distance")
        elif model_label == "RF":
            model = RandomForestRegressor(
                random_state=seed,
                n_estimators=400,
                max_features=2,
                min_samples_leaf=1,
                min_samples_split=2
            )
        elif model_label == "DT":
            model = DecisionTreeRegressor(
                random_state=seed,
                splitter="best",
                max_features=2,
                min_samples_leaf=1,
                min_samples_split=2
            )
        elif model_label == "NN1":
            model = MLPRegressor(
                random_state=seed,
                max_iter=5000,
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.001,
                hidden_layer_sizes=(int(y.shape[0] * 0.1))
            )
        elif model_label == "NN2":
            model = MLPRegressor(
                random_state=seed,
                max_iter=5000,
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.001,
                hidden_layer_sizes=(int(y.shape[0] * 0.5), int(y.shape[0] * 0.2))
            )
        elif model_label == "NN3":
            model = MLPRegressor(
                random_state=seed,
                max_iter=5000,
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.001,
                hidden_layer_sizes=(
                    int(y.shape[0] * 0.5), int(y.shape[0] * 0.2), int(y.shape[0] * 0.1)
                )
            )

        # Print model config
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("Training RocML model without tuning:")
        print(f"    program: {program}")
        print(f"    sample: {sample_id}")
        print(f"    model: {model_label_full}")
        print(f"    k folds: {kfolds}")
        print(f"    features:")
        print(f"        Pressure (GPa)")
        print(f"        Temperature (K)")
        print(f"    targets:")
        for param, unit in zip(parameters, units_labels):
            print(f"        {param} {unit}")
        print(f"    features dataset shape: ({W}, {W}, {X.shape[1]})")
        print(f"    targets dataset shape: ({W}, {W}, {y.shape[1]})")
        print(f"    training dataset NANs: {n_nans_training}")
        print(f"    hyperparameters: no-tune (predefined)")
        print("+++++++++++++++++++++++++++++++++++++++++++++")

    # Define ML model and grid search param space for hyperparameter tuning
    else:
        # K-fold cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=seed)

        if model_label == "KN":
            model = KNeighborsRegressor()
            param_grid = dict(
                n_neighbors=[2, 4, 8],
                weights=["uniform", "distance"]
            )
        elif model_label == "RF":
            model = RandomForestRegressor(random_state=seed)
            param_grid = dict(
                n_estimators=[400, 800, 1200],
                max_features=[1, 2, 3],
                min_samples_leaf=[1, 2, 3],
                min_samples_split=[2, 4, 6]

            )
        elif model_label == "DT":
            model = DecisionTreeRegressor(random_state=seed)
            param_grid = dict(
                splitter=["best", "random"],
                max_features=[1, 2, 3],
                min_samples_leaf=[1, 2, 3],
                min_samples_split=[2, 4, 6]

            )
        elif model_label == "NN1":
            model = MLPRegressor(
                random_state=seed,
                max_iter=5000,
                activation="relu",
                alpha=0.001
            )
            param_grid = dict(
                hidden_layer_sizes=[
                    (int(y.shape[0] * 0.1)),
                    (int(y.shape[0] * 0.2)),
                    (int(y.shape[0] * 0.5))
                ]
            )
        elif model_label == "NN2":
            model = MLPRegressor(
                random_state=seed,
                max_iter=5000,
                activation="relu",
                alpha=0.001
            )
            param_grid = dict(
                hidden_layer_sizes=[
                    (int(y.shape[0] * 0.1), int(y.shape[0] * 0.2)),
                    (int(y.shape[0] * 0.2), int(y.shape[0] * 0.2)),
                    (int(y.shape[0] * 0.5), int(y.shape[0] * 0.2))
                ]
            )
        elif model_label == "NN3":
            model = MLPRegressor(
                random_state=seed,
                max_iter=5000,
                activation="relu",
                alpha=0.001
            )
            param_grid = dict(
                hidden_layer_sizes=[
                    (int(y.shape[0] * 0.1), int(y.shape[0] * 0.2), int(y.shape[0] * 0.1)),
                    (int(y.shape[0] * 0.2), int(y.shape[0] * 0.2), int(y.shape[0] * 0.1)),
                    (int(y.shape[0] * 0.5), int(y.shape[0] * 0.2), int(y.shape[0] * 0.1))
                ]
            )

        # Perform grid search hyperparameter tuning
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=nprocs,
            verbose=1
        )
        grid_search.fit(X_scaled, y_scaled)

        # Define ML model with tuned hyperparameters
        if model_label == "KN":
            model = KNeighborsRegressor(
                n_neighbors=grid_search.best_params_["n_neighbors"],
                weights=grid_search.best_params_["weights"]
            )
        elif model_label == "RF":
            model = RandomForestRegressor(
                random_state=seed,
                n_estimators=grid_search.best_params_["n_estimators"],
                max_features=grid_search.best_params_["max_features"],
                min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                min_samples_split=grid_search.best_params_["min_samples_split"]
            )
        elif model_label == "DT":
            model = DecisionTreeRegressor(
                random_state=seed,
                splitter=grid_search.best_params_["splitter"],
                max_features=grid_search.best_params_["max_features"],
                min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                min_samples_split=grid_search.best_params_["min_samples_split"]
            )
        elif model_label in ["NN1", "NN2", "NN3"]:
            model = MLPRegressor(
                random_state=seed,
                max_iter=5000,
                activation=grid_search.best_params_["activation"],
                alpha=grid_search.best_params_["alpha"],
                learning_rate_init=grid_search.best_params_["learning_rate_init"],
                hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"]
            )

        # Print model config
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("Training RocML model with grid search tuning:")
        print(f"    program: {program}")
        print(f"    sample: {sample_id}")
        print(f"    model: {model_label_full}")
        print(f"    k folds: {kfolds}")
        print(f"    features:")
        print(f"        Pressure (GPa)")
        print(f"        Temperature (K)")
        print(f"    targets:")
        for param, unit in zip(parameters, units_labels):
            print(f"        {param} {unit}")
        print(f"    training dataset shape: ({W}, {W}, {n_features + 2})")
        print(f"    training dataset NANs: {n_nans_training}")
        print(f"    features shape: {X.shape}")
        print(f"    targets shape: {y.shape}")
        print(f"    hyperparameters:")
        for key, value in grid_search.best_params_.items():
            print(f"        {key}: {value}")
        print("+++++++++++++++++++++++++++++++++++++++++++++")

    # K-fold cross-validation
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

    # Iterate over K-fold cross-validations in parallel
    # Combine the data and parameters needed for each fold into a list
    fold_data_list = [
        (
            (train_index, test_index),
            X_scaled,
            y_scaled,
            X_scaled_valid,
            y_scaled_valid,
            model,
            scaler_X,
            scaler_y,
            scaler_X_valid,
            scaler_y_valid
        ) for fold_idx, (train_index, test_index) in enumerate(kf.split(X))
    ]

    # Initialize the pool of processes
    with Pool(processes=nprocs) as pool:
        results = pool.map(process_fold, fold_data_list)

        # Wait for all processes
        pool.close()
        pool.join()

    # Unpack the results from the parallel kfolds
    rmse_test_scores = []
    r2_test_scores = []
    rmse_valid_scores = []
    r2_valid_scores = []
    training_times = []
    inference_times = []

    for (rmse_test, r2_test, rmse_valid, r2_valid, training_time, inference_time) in results:
        rmse_test_scores.append(rmse_test)
        r2_test_scores.append(r2_test)
        rmse_valid_scores.append(rmse_valid)
        r2_valid_scores.append(r2_valid)
        training_times.append(training_time)
        inference_times.append(inference_time)

    # Stack arrays
    rmse_test_scores = np.stack(rmse_test_scores)
    r2_test_scores = np.stack(r2_test_scores)
    rmse_valid_scores = np.stack(rmse_valid_scores)
    r2_valid_scores = np.stack(r2_valid_scores)

    # Calculate performance values with uncertainties
    rmse_test_mean = np.mean(rmse_test_scores, axis=0)
    rmse_test_std = np.std(rmse_test_scores, axis=0)
    r2_test_mean = np.mean(r2_test_scores, axis=0)
    r2_test_std = np.std(r2_test_scores, axis=0)
    rmse_valid_mean = np.mean(rmse_valid_scores, axis=0)
    rmse_valid_std = np.std(rmse_valid_scores, axis=0)
    r2_valid_mean = np.mean(r2_valid_scores, axis=0)
    r2_valid_std = np.std(r2_valid_scores, axis=0)
    training_time_mean = np.mean(training_times)
    training_time_std = np.std(training_times)
    inference_time_mean = np.mean(inference_times)
    inference_time_std = np.std(inference_times)

    # Config and performance info
    model_info = {
        "sample": sample_id,
        "model": model_label,
        "program": program,
        "grid": (W-1)**2,
        "n_params": len(parameters),
        "k_folds": kfolds,
        "training_time_mean": round(training_time_mean, 3),
        "training_time_std": round(training_time_std, 3),
        "inference_time_mean": round(inference_time_mean, 3),
        "inference_time_std": round(inference_time_std, 3)
    }

    # Add performance metrics for each parameter to the dictionary
    for i, param in enumerate(parameters):
        model_info[f"rmse_test_mean_{param}"] = round(rmse_test_mean[i], 3)
        model_info[f"rmse_test_std_{param}"] = round(rmse_test_std[i], 3)
        model_info[f"r2_test_mean_{param}"] = round(r2_test_mean[i], 3),
        model_info[f"r2_test_std_{param}"] = round(r2_test_std[i], 3),
        model_info[f"rmse_valid_mean_{param}"] = round(rmse_valid_mean[i], 3),
        model_info[f"rmse_valid_std_{param}"] = round(rmse_valid_std[i], 3),
        model_info[f"r2_valid_mean_{param}"] = round(r2_valid_mean[i], 3),
        model_info[f"r2_valid_std_{param}"] = round(r2_valid_std[i], 3)

    # Print performance
    print(f"{model_label_full} performance:")
    print(f"    training time: {training_time_mean:.3f} ± {training_time_std:.3f}")
    print(f"    inference time: {inference_time_mean:.3f} ± {inference_time_std:.3f}")
    print(f"    rmse test (%):")
    for r, e, p in zip(rmse_test_mean, rmse_test_std, parameters):
        print(f"        {p}: {r:.3f} ± {e:.3f}")
    print(f"    r2 test:")
    for r, e, p in zip(r2_test_mean, r2_test_std, parameters):
        print(f"        {p}: {r:.3f} ± {e:.3f}")
    print(f"    rmse valid (%):")
    for r, e, p in zip(rmse_valid_mean, rmse_valid_std, parameters):
        print(f"        {p}: {r:.3f} ± {e:.3f}")
    print(f"    r2 valid:")
    for r, e, p in zip(r2_valid_mean, r2_valid_std, parameters):
        print(f"        {p}: {r:.3f} ± {e:.3f}")
    print("+++++++++++++++++++++++++++++++++++++++++++++")

    # Train model for plotting
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_scaled,
        test_size=0.2,
        random_state=seed
    )

    # Train ML model
    model.fit(X_train, y_train)

    # Reshape and scale validation dataset features
    X_valid = features_array_valid.reshape(W*W, 2)
    X_scaled_valid = scaler_X_valid.fit_transform(X_valid)

    # Make predictions on validation dataset
    valid_pred_scaled = model.predict(X_scaled_valid)

    # Inverse transform predictions
    valid_pred_original = scaler_y_valid.inverse_transform(valid_pred_scaled)

    # Reshape the predictions to match the grid shape for visualization
    valid_pred_array_original = valid_pred_original.reshape(W, W, n_features)

    for i, param in enumerate(parameters):
        # Make nans consistent
        mask = np.isnan(targets_array_valid[:,:,i])
        valid_pred_array_original[:,:,i][mask] = np.nan

        # Compute normalized diff
        diff_norm = (
            (targets_array_valid[:,:,i] - valid_pred_array_original[:,:,i]) /
            ((targets_array_valid[:,:,i] + valid_pred_array_original[:,:,i]) / 2) * 100
        )

        # Make nans consistent
        diff_norm[mask] = np.nan

        # Plot training data distribution and ML model predictions
        colormap = plt.cm.get_cmap("tab10")

        # Reverse color scale
        if palette in ["grey"]:
            color_reverse = False
        else:
            color_reverse = True

        # Plot target array
        visualize_2d_pt_grid(
            features_array[:,:,0],
            features_array[:,:,1],
            targets_array[:,:,i],
            param,
            title=f"{program}",
            palette=palette,
            color_discrete=False,
            color_reverse=color_reverse,
            vmin=vmin[i],
            vmax=vmax[i],
            filename=f"{filename}-{param}-targets.png",
            fig_dir=fig_dir
        )

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            features_array[:,:,1],
            features_array[:,:,0],
            targets_array[:,:,i],
            cmap=cmap,
            vmin=vmin[i],
            vmax=vmax[i]
        )
        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.set_zlim(vmin[i] - (vmin[i] * 0.05), vmax[i] + (vmax[i] * 0.05))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(f"{program}", y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")
        cbar = fig.colorbar(
            surf,
            ax=ax,
            ticks=np.linspace(vmin[i], vmax[i], num=4),
            label="",
            shrink=0.6
        )
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        cbar.ax.set_ylim(vmin[i], vmax[i])
        plt.savefig(f"{fig_dir}/{filename}-{param}-targets-surf.png")
        plt.close()

        # Plot ML model predictions array
        visualize_2d_pt_grid(
            features_array_valid[:,:,0],
            features_array_valid[:,:,1],
            valid_pred_array_original[:,:,i],
            param,
            title=f"{model_label_full}",
            palette=palette,
            color_discrete=False,
            color_reverse=color_reverse,
            vmin=vmin[i],
            vmax=vmax[i],
            filename=f"{filename}-{param}-predictions.png",
            fig_dir=fig_dir
        )

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            features_array_valid[:,:,1],
            features_array_valid[:,:,0],
            valid_pred_array_original[:,:,i],
            cmap=cmap,
            vmin=vmin[i],
            vmax=vmax[i]
        )
        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.set_zlim(vmin[i] - (vmin[i] * 0.05), vmax[i] + (vmax[i] * 0.05))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(f"{model_label_full}", y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")
        cbar = fig.colorbar(
            surf,
            ax=ax,
            ticks=np.linspace(vmin[i], vmax[i], num=4),
            label="",
            shrink=0.6
        )
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        cbar.ax.set_ylim(vmin[i], vmax[i])
        plt.savefig(f"{fig_dir}/{filename}-{param}-surf.png")
        plt.close()

        # Plot PT normalized diff targets vs. ML model predictions
        visualize_2d_pt_grid(
            features_array_valid[:,:,0],
            features_array_valid[:,:,1],
            diff_norm,
            param,
            title="Percent Difference",
            palette="seismic",
            color_discrete=False,
            color_reverse=False,
            vmin=vmin[i],
            vmax=vmax[i],
            filename=f"{filename}-{param}-diff.png",
            fig_dir=fig_dir
        )

        # 3D surface
        vmin_diff = -np.max(np.abs(diff_norm[np.logical_not(np.isnan(diff_norm))]))
        vmax_diff = np.max(np.abs(diff_norm[np.logical_not(np.isnan(diff_norm))]))

        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            features_array_valid[:,:,1],
            features_array_valid[:,:,0],
            diff_norm,
            cmap="seismic",
            vmin=vmin_diff,
            vmax=vmax_diff
        )
        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.set_zlim(vmin_diff - (vmin_diff * 0.05), vmax_diff + (vmax_diff * 0.05))
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(f"Percent Difference", y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")
        cbar = fig.colorbar(
            surf,
            ax=ax,
            ticks=[vmin_diff, vmin_diff/2, 0, vmax_diff/2, vmax_diff],
            label="",
            shrink=0.6
        )
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        cbar.ax.set_ylim(vmin_diff, vmax_diff)
        plt.savefig(f"{fig_dir}/{filename}-{param}-diff-surf.png")
        plt.close()

        # Reshape results and transform units for MAGEMin
        if program == "MAGEMin":
            results_mgm = {
                "P": [P * 10 for P in features_array[:,:,0].ravel().tolist()],
                "T": [T - 273 for T in features_array[:,:,1].ravel().tolist()],
                param: targets_array[:,:,i].ravel().tolist()
            }

            results_ppx = None

            if param == "DensityOfFullAssemblage":
                results_mgm[param] = [x * 1000 for x in results_mgm[param]]

        # Reshape results and transform units for Perple_X
        if program == "Perple_X":
            results_ppx = {
                "P": [P * 10 for P in features_array[:,:,0].ravel().tolist()],
                "T": [T - 273 for T in features_array[:,:,1].ravel().tolist()],
                param: targets_array[:,:,i].ravel().tolist()
            }

            results_mgm = None

            if param == "DensityOfFullAssemblage":
                results_ppx[param] = [x * 1000 for x in results_ppx[param]]

        # Reshape results and transform units for ML model
        results_rocml = {
            "P": [P * 10 for P in features_array_valid[:,:,0].ravel().tolist()],
            "T": [T - 273 for T in features_array_valid[:,:,1].ravel().tolist()],
            param: valid_pred_array_original[:,:,i].ravel().tolist()
        }

        if param == "DensityOfFullAssemblage":
            results_rocml[param] = [x * 1000 for x in results_rocml[param]]

        # Plot PREM comparisons
        metrics = [rmse_valid_mean[i], r2_valid_mean[i]]

        # Set geotherm threshold for extracting depth profiles
        res = W - 1
        if res <= 8:
            geotherm_threshold = 4
        elif res <= 16:
            geotherm_threshold = 2
        elif res <= 32:
            geotherm_threshold = 1
        elif res <= 64:
            geotherm_threshold = 0.5
        elif res <= 128:
            geotherm_threshold = 0.25
        else:
            geotherm_threshold = 0.125

        if param == "DensityOfFullAssemblage":
            visualize_PREM(
                "assets/data/prem.csv",
                param,
                "g/cm$^3$",
                results_mgm=results_mgm,
                results_ppx=results_ppx,
                results_rocml=results_rocml,
                model=f"{model_label}",
                geotherm_threshold=geotherm_threshold,
                depth=True,
                metrics=metrics,
                title=f"{model_label_full}",
                figwidth=figwidth,
                filename=f"{filename}-{param}-prem.png",
                fig_dir=fig_dir
            )

        if param in ["Vp", "Vs"]:
            visualize_PREM(
                "assets/data/prem.csv",
                param,
                "km/s",
                results_mgm=results_mgm,
                results_ppx=results_ppx,
                results_rocml=results_rocml,
                model=f"{model_label}",
                geotherm_threshold=geotherm_threshold,
                depth=True,
                metrics=metrics,
                title=f"{model_label_full}",
                figwidth=figwidth,
                filename=f"{filename}-{param}-prem.png",
                fig_dir=fig_dir
            )

    return model, model_info

# Extract features array from training dataset
def extract_features(results):
    """
    """
    # Get PT values and transform units
    P = [P / 10 for P in results["P"]]
    T = [T + 273 for T in results["T"]]

    # Reshape into (W, 1) arrays
    P_array = np.unique(np.array(P)).reshape(-1, 1)
    T_array = np.unique(np.array(T)).reshape(1, -1)

    # Get array dimensions
    W = P_array.shape[0]

    # Reshape into (W, W) arrays by repeating values
    P_grid = np.tile(P_array, (1, W))
    T_grid = np.tile(T_array, (W, 1))

    # Combine P and T grids into a single feature dataset with shape (W, W, 2)
    features_array = np.stack((P_grid, T_grid), axis=-1)

    return P, T, features_array

# Train RocMLs
def train_rocml(
        sample_id,
        res,
        parameters,
        mask_geotherm=False,
        magemin=True,
        perplex=True,
        model="DT",
        tune=False,
        kfolds=cpu_count()-2,
        parallel=True,
        nprocs=cpu_count()-2,
        seed=42,
        palette="bone",
        out_dir=f"{os.getcwd()}/runs",
        fig_dir=f"{os.getcwd()}/figs",
        data_dir=f"{os.getcwd()}/assets/data"):
    """
    Perform ML model regression.

    Parameters:
        sample_id (str): The identifier of the sample for which GFEM results are to be
                         analyzed.
        parameters (list): A list of parameters for which ML models will be performed.
                           Supported parameters depend on the GFEM program used.
        seed (int, optional): Random seed for reproducibility. Default is 42.
        palette (str, optional): The color palette for the plots. Default is "bone".
        out_dir (str, optional): Directory to store intermediate output files from the GFEM
                                 program. Default is "./runs".
        fig_dir (str, optional): Directory to save the generated visualization figures.
                                 Default is "./figs".
        data_dir (str, optional): Directory containing data files needed for visualization.
                                  Default is "./assets/data".

    Notes:
        - This function retrieves GFEM results from the specified program and sample.
        - It preprocesses the feature array (P and T) and the targets array for ML model
          analysis.
          specified parameter.
        - The results of ML model, including plots and performance information, are saved to
          appropriate files.
    """
    if magemin:
        # Get results
        results_mgm = process_magemin_grid(sample_id, "train", res, out_dir)
        results_mgm_valid = process_magemin_grid(sample_id, "valid", res, out_dir)

        # Get features arrays
        P_mgm, T_mgm, features_mgm = extract_features(results_mgm)
        P_mgm_valid, T_mgm_valid, features_mgm_valid = extract_features(results_mgm_valid)

    if perplex:
        # Get results
        results_ppx = process_perplex_grid(
            f"{out_dir}/{sample_id}/perplex_train_{res}/{sample_id}_grid.tab",
            f"{out_dir}/{sample_id}/perplex_train_{res}/{sample_id}_assemblages.txt"
        )
        results_ppx_valid = process_perplex_grid(
            f"{out_dir}/{sample_id}/perplex_valid_{res}/{sample_id}_grid.tab",
            f"{out_dir}/{sample_id}/perplex_valid_{res}/{sample_id}_assemblages.txt"
        )

        # Make liquid fraction consistent
        results_ppx["LiquidFraction"] = [
            0.0 if np.isnan(x) else x for x in results_ppx["LiquidFraction"]
        ]
        results_ppx_valid["LiquidFraction"] = [
            0.0 if np.isnan(x) else x for x in results_ppx_valid["LiquidFraction"]
        ]

        # Get features arrays
        P_ppx, T_ppx, features_ppx = extract_features(results_ppx)
        P_ppx_valid, T_ppx_valid, features_ppx_valid = extract_features(results_ppx_valid)

    # Create empty lists
    units = []
    targets_mgm, targets_mgm_valid, targets_ppx, targets_ppx_valid = [], [], [], []
    vmin, vmax, vmin_valid, vmax_valid = [], [], [], []

    for i, parameter in enumerate(parameters):
        # Units
        if parameter == "DensityOfFullAssemblage":
            units.append("g/cm$^3$")
        if parameter in ["Vp", "Vs"]:
            units.append("km/s")
        if parameter == "LiquidFraction":
            units.append("%")

        if magemin:
            # Target array with shape (W, W)
            targets_mgm.append(create_PT_grid(
                P_mgm,
                T_mgm,
                results_mgm[parameter],
                mask_geotherm
            ))
            targets_mgm_valid.append(create_PT_grid(
                P_mgm_valid,
                T_mgm_valid,
                results_mgm_valid[parameter],
                mask_geotherm
            ))

        if perplex:
            # Target array with shape (W, W)
            targets_ppx.append(create_PT_grid(
                P_ppx,
                T_ppx,
                results_ppx[parameter],
                mask_geotherm
            ))
            # Target array with shape (W, W)
            targets_ppx_valid.append(create_PT_grid(
                P_ppx_valid,
                T_ppx_valid,
                results_ppx_valid[parameter],
                mask_geotherm
            ))

        # Get min max of target array to plot colorbars on the same scales
        if magemin and not perplex:
            vmin.append(np.min(np.abs(
                targets_mgm[i][np.logical_not(np.isnan(targets_mgm[i]))]
            )))
            vmax.append(np.max(np.abs(
                targets_mgm[i][np.logical_not(np.isnan(targets_mgm[i]))]
            )))
            vmin_valid.append(np.min(np.abs(
                targets_mgm_valid[i][np.logical_not(np.isnan(targets_mgm_valid[i]))]
            )))
            vmax_valid.append(np.max(np.abs(
                targets_mgm_valid[i][np.logical_not(np.isnan(targets_mgm_valid[i]))]
            )))

        if perplex and not magemin:
            vmin.append(np.min(np.abs(
                targets_ppx[i][np.logical_not(np.isnan(targets_ppx[i]))]
            )))
            vmax.append(np.max(np.abs(
                targets_ppx[i][np.logical_not(np.isnan(targets_ppx[i]))]
            )))
            vmin_valid.append(np.min(np.abs(
                targets_ppx_valid[i][np.logical_not(np.isnan(targets_ppx_valid[i]))]
            )))
            vmax_valid.append(np.max(np.abs(
                targets_ppx_valid[i][np.logical_not(np.isnan(targets_ppx_valid[i]))]
            )))

        if magemin and perplex:
            vmin.append(min(
                np.min(np.abs(targets_mgm[i][np.logical_not(np.isnan(targets_mgm[i]))])),
                np.min(np.abs(targets_ppx[i][np.logical_not(np.isnan(targets_ppx[i]))]))
            ))
            vmax.append(max(
                np.max(np.abs(targets_mgm[i][np.logical_not(np.isnan(targets_mgm[i]))])),
                np.max(np.abs(targets_ppx[i][np.logical_not(np.isnan(targets_ppx[i]))]))
            ))
            vmin_valid.append(min(
                np.min(np.abs(targets_mgm_valid[i][
                    np.logical_not(np.isnan(targets_mgm_valid[i]))
                ])),
                np.min(np.abs(targets_ppx_valid[i][
                    np.logical_not(np.isnan(targets_ppx_valid[i]))
                ]))
            ))
            vmax_valid.append(max(
                np.max(np.abs(targets_mgm_valid[i][
                    np.logical_not(np.isnan(targets_mgm_valid[i]))
                ])),
                np.max(np.abs(targets_ppx_valid[i][
                    np.logical_not(np.isnan(targets_ppx_valid[i]))
                ]))
            ))

        # Transform units
        if parameter == "DensityOfFullAssemblage":
            vmin[i] = vmin[i] / 1000
            vmax[i] = vmax[i] / 1000
            vmin_valid[i] = vmin_valid[i] / 1000
            vmax_valid[i] = vmax_valid[i] / 1000

        if parameter == "LiquidFraction":
            vmin[i] = vmin[i] * 100
            vmax[i] = vmax[i] * 100
            vmin_valid[i] = vmin_valid[i] * 100
            vmax_valid[i] = vmax_valid[i] * 100

        # Change model string for filename
        model_label = model.replace(" ", "-")

    # Combine target arrays for all parameters
    if magemin:
        targets_mgm = np.stack(targets_mgm, -1)
        targets_mgm_valid = np.stack(targets_mgm_valid, -1)

    if perplex:
        targets_ppx = np.stack(targets_ppx, -1)
        targets_ppx_valid = np.stack(targets_ppx_valid, -1)

    # Train models, predict, analyze
    if magemin:
        model_mgm, info_mgm = cv_rocml(
            features_mgm, targets_mgm, features_mgm_valid,
            targets_mgm_valid, parameters, units, "MAGEMin", sample_id,
            model, tune, seed, kfolds, parallel, nprocs, vmin, vmax,
            filename=f"MAGEMin-{sample_id}-{model_label}", fig_dir=fig_dir
        )

        # Write ML model config and performance info to csv
        append_to_csv("assets/data/benchmark-rocmls-performance.csv", info_mgm)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if perplex:
        model_ppx, info_ppx = cv_rocml(
            features_ppx, targets_ppx, features_ppx_valid,
            targets_ppx_valid, parameters, units, "Perple_X", sample_id,
            model, tune, seed, kfolds, parallel, nprocs, vmin, vmax,
            filename=f"Perple_X-{sample_id}-{model_label}", fig_dir=fig_dir
        )

        # Write ML model config and performance info to csv
        append_to_csv("assets/data/benchmark-rocmls-performance.csv", info_ppx)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print("=============================================")

# Visualize rocml performance
def visualize_rocml_performance(
        datafile="assets/data/benchmark-rocmls-performance.csv",
        benchmark_times="assets/data/benchmark-gfem-efficiency.csv",
        sample_id="PUM",
        parameter="DensityOfFullAssemblage",
        res=128,
        palette="tab10",
        fontsize=22,
        figwidth=6.3,
        figheight=4.725,
        filename="rocml",
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Visualize regression metrics using a facet barplot.

    Parameters:
        datafile (str): The path to the CSV file containing the data to be visualized.
        palette (str, optional): The color palette to use for plotting the bars. Default is
                                 "tab10".
        fontsize (int, optional): The font size for the plot's text. Default is 12.
        figwidth (float, optional): The width of the plot figure in inches. Default is 8.3.
        figheight (float, optional): The height of the plot figure in inches. Default is 3.8.
        filename (str, optional): The filename to save the plot as an image. Default is
                                  "regression-metrics.png".
        fig_dir (str, optional): The directory path to save the plot image. Default is a
                                 "figs" directory within the current working directory.

    Returns:
        None: The function saves the plot as an image file or displays it if `filename` is
              not provided.

    Note:
        This function reads the data from a CSV file, extracts specific regression metrics,
        and creates a facet barplot for side-by-side comparison of the metrics across
        different models and programs.

        The CSV file should contain columns "model", "program", "training_time_mean",
        "inference_time_mean", "rmse_valid_mean", and "units". The "units" column is used to
        label the y-axis of the RMSE plot.

        The function uses Matplotlib for plotting and sets various plot style and settings to
        enhance the visualization.

        The color palette, font size, and plot dimensions can be customized using the
        respective parameters.

        The y-axis limits for each metric can also be customized in the `y_limits` dictionary
        within the function.

        If the `filename` parameter is provided, the plot will be saved as an image file with
        the specified name in the `fig_dir` directory. Otherwise, the plot will be displayed
        directly on the screen.
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read regression data
    data = pd.read_csv(datafile)
    data = data[data["sample"] == sample_id]
    data = data[data["grid"] == res**2]

    # Summarize data
    numeric_columns = data.select_dtypes(include=[float, int]).columns
    summary_df = data.groupby("model")[numeric_columns].mean().reset_index()

    # Get MAGEMin and Perple_X benchmark times
    benchmark_times = pd.read_csv(benchmark_times)

    filtered_times = benchmark_times[
        (benchmark_times["sample"] == sample_id) & (benchmark_times["grid"] == res**2)
    ]

    time_mgm = np.mean(
        filtered_times[filtered_times["program"] == "magemin"]["time"].values /
        filtered_times[filtered_times["program"] == "magemin"]["grid"].values *
        1000
    )
    time_ppx = np.mean(
        filtered_times[filtered_times["program"] == "perplex"]["time"].values /
        filtered_times[filtered_times["program"] == "perplex"]["grid"].values *
        1000
    )

    # Define the metrics to plot
    metrics = [
        f"training_time_mean",
        f"inference_time_mean",
        f"rmse_test_mean_{parameter}",
        f"rmse_valid_mean_{parameter}"
    ]
    metric_names = [
        "Training Efficiency",
        "Prediction Efficiency",
        "Training Error",
        "Validation Error"
    ]

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Define the colors for the programs
    colormap = plt.cm.get_cmap(palette)
    models = ["KN", "DT", "RF", "NN1", "NN2", "NN3"]

    # Create a dictionary to map each model to a specific color
    color_mapping = {
        "KN": colormap(2),
        "DT": colormap(0),
        "RF": colormap(4),
        "NN1": colormap(1),
        "NN2": colormap(3),
        "NN3": colormap(5)
    }

    # Get the corresponding colors for each model
    colors = [color_mapping[model] for model in models]

    # Loop through each metric and create a subplot
    for i, metric in enumerate(metrics):
        # Create the facet barplot
        plt.figure(figsize=(figwidth, figheight))

        # Define the offset for side-by-side bars
        bar_width = 0.45

        # Get order of sorted bars
        order = summary_df[metric].sort_values().index
        models_order = summary_df.loc[order]["model"].tolist()

        # Bar positions
        x_positions = np.arange(len(summary_df[metric]))

        # Plot the bars for each program
        bars = plt.bar(
            x_positions * bar_width,
            summary_df.loc[order][metric],
            edgecolor="black",
            color=[color_mapping[model] for model in models_order],
            label=models_order if i == 1 else "",
            width=bar_width
        )

        # Show MAGEMin and Perple_X compute times
        if metric == f"inference_time_mean":
            mgm_line = plt.axhline(time_mgm, color="black", linestyle="-", label="MAGEMin")
            ppx_line = plt.axhline(time_ppx, color="black", linestyle="--", label="Perple_X")

        plt.gca().set_xticks([])
        plt.gca().set_xticklabels([])

        # Plot titles
        if metric == f"training_time_mean":
            plt.title(f"{metric_names[i]}")
            plt.ylabel(f"Elapsed Time (ms)")
            plt.yscale("log")
        elif metric == f"inference_time_mean":
            plt.title(f"{metric_names[i]}")
            plt.ylabel(f"Elapsed Time (ms)")
            plt.yscale("log")
            handles = [mgm_line, ppx_line]
            labels = [handle.get_label() for handle in handles]
            legend = plt.legend(fontsize="x-small")
            legend.set_bbox_to_anchor((0, 0.94))
        elif metric == f"rmse_test_mean_{parameter}":
            plt.errorbar(
                x_positions * bar_width,
                summary_df.loc[order][f"rmse_test_mean_{parameter}"],
                yerr=summary_df.loc[order][f"rmse_test_std_{parameter}"] * 2,
                fmt="none",
                capsize=5,
                color="black",
                linewidth=2
            )
            plt.title(f"{metric_names[i]}")
            plt.ylabel("RMSE (%)")
            plt.ylim(bottom=0)
            plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        elif metric == f"rmse_valid_mean_{parameter}":
            plt.errorbar(
                x_positions * bar_width,
                summary_df.loc[order][f"rmse_valid_mean_{parameter}"],
                yerr=summary_df.loc[order][f"rmse_valid_std_{parameter}"] * 2,
                fmt="none",
                capsize=5,
                color="black",
                linewidth=2
            )
            plt.title(f"{metric_names[i]}")
            plt.ylabel("RMSE (%)")
            plt.ylim(bottom=0)
            plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(
                f"{fig_dir}/{filename}-{metric.replace('_', '-')}-{sample_id}-{res}.png"
            )
        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()