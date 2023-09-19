#######################################################
##                  Load Libraries                   ##
#######################################################

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
import datetime
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
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from matplotlib.ticker import FixedLocator
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

#######################################################
##      General Helper Functions for Scripting       ##
#######################################################

# Print session info for logging
def print_session_info(conda_file=None, makefile=None):
    """
    Print session information for logging.

    Parameters:
        conda_file (str, optional): The path to the Conda YAML file. Defaults to None.
        makefile (str, optional): The path to the Makefile. Defaults to None.

    This function prints session information for logging purposes, including:
    - Python version.
    - Loaded package versions (if a Conda file is provided).
    - Operating system information.
    - Random seed (if a Makefile is provided and contains a "SEED" variable).

    If Conda and Makefile paths are not provided, it prints appropriate messages.
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

# Read makefile variables
def read_makefile_variable(makefile, variable):
    """
    Read a variable's value from a Makefile.

    Parameters:
        makefile (str): The path to the Makefile.
        variable (str): The name of the variable to retrieve.

    Returns:
        str or None: The value of the specified variable, or None if the variable is not found.

    This function reads a Makefile and searches for a specific variable by name.
    If the variable is found, its value is extracted and returned as a string.
    If the variable is not found or if there is an error reading the Makefile, None is returned.
    """
    try:
        with open(makefile, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.strip().startswith(variable):
                    return line.split("=")[1].strip()
    except IOError as e:
        print(f"Error reading Makefile: {e}")
    return None

# Read conda packages from yaml
def get_conda_packages(conda_file):
    """
    Read Conda packages from a YAML file.

    Parameters:
        conda_file (str): The path to the Conda YAML file.

    Returns:
        list: A list of Conda package dependencies specified in the YAML file.

    Raises:
        IOError: If there was an error reading the Conda file.
        yaml.YAMLError: If there was an error parsing the YAML file.

    This function reads a Conda YAML file and extracts the list of Conda package
    dependencies specified in the "dependencies" section of the file. It returns
    this list of packages.

    If any errors occur during file reading or parsing, appropriate exceptions are
    raised and an empty list is returned.
    """
    try:
        with open(conda_file, "r") as file:
            conda_data = yaml.safe_load(file)
        return conda_data.get("dependencies", [])
    except (IOError, yaml.YAMLError) as e:
        print(f"Error reading Conda file: {e}")
        return []

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
        "--source",
        type=str,
        help="Specify the source argument ...",
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
        "--oxides",
        type=parse_list_of_strings,
        help="Specify the oxides argument ...",
        required=False
    )
    parser.add_argument(
        "--npca",
        type=int,
        help="Specify the npca argument ...",
        required=False
    )
    parser.add_argument(
        "--kcluster",
        type=int,
        help="Specify the kcluster argument ...",
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
        "--configdir",
        type=str,
        help="Specify the configdir argument ...",
        required=False
    )
    parser.add_argument(
        "--perplexdir",
        type=str,
        help="Specify the perplexdir argument ...",
        required=False
    )
    parser.add_argument(
        "--logfile",
        type=str,
        help="Specify the logfile argument ...",
        required=False
    )
    parser.add_argument(
        "--emsonly",
        type=str,
        help="Specify the emsonly argument ...",
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
    parser.add_argument(
        "--verbose",
        type=str,
        help="Specify the verbose argument ...",
        required=False
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the parallel argument to a boolean value based on the string value
    parallel = args.parallel.lower() == "true" if args.parallel else False
    emsonly = args.emsonly.lower() == "true" if args.emsonly else False
    verbose = args.verbose.lower() == "true" if args.verbose else False

    # Assign the modified parallel value back to the args object
    args.parallel = parallel
    args.emsonly = emsonly
    args.verbose = verbose

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
    emsonly = args.emsonly
    comp = args.comp
    frac = args.frac
    source = args.source
    sampleid = args.sampleid
    params = args.params
    models = args.models
    tune = args.tune
    normox = args.normox
    oxides = args.oxides
    npca = args.npca
    kcluster = args.kcluster
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
    configdir = args.configdir
    perplexdir = args.perplexdir
    logfile = args.logfile
    verbose = args.verbose

    # MAGEMin oxide options
    oxide_list_magemin = [
        "SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O",
        "NA2O", "TIO2", "FE2O3", "CR2O3", "H2O"
    ]

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

        print(f"    dataset resolution: {res}")

        if Tmin is not None and Tmax is not None:
            print(f"    Trange: [{Tmin}, {Tmax}, {res}]")

        if Pmin is not None and Pmax is not None:
            print(f"    Prange: [{Pmin}, {Pmax}, {res}]")

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

    if source is not None:
        print(f"    samples datafile: {source}")

        valid_args["source"] = source

    if sampleid is not None:
        if source is not None:
            # Get sample composition
            sample_comp = get_sample_composition(source, sampleid)

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

        if sample_comp is not None:
            # Normalize composition
            sample_norm = normalize_sample(sample=sample_comp, components=normox)

            print("    Normalized composition:")

            for component, value in zip(oxide_list_magemin, sample_norm):
                formatted_value = "{:.3f}".format(value)
                print(f"        {component}: {formatted_value}")

        valid_args["normox"] = normox

    if oxides is not None:
        if check_non_matching_strings(oxides, oxide_list_magemin[:-1]):
            raise ValueError(
                "Invalid --oxides argument ...\n"
                f"Can only use oxides {oxide_list_magemin[:-1]}"
            )

        print(f"    Selected oxides:")

        for oxide in oxides:
            print(f"        {oxide}")

        valid_args["oxides"] = oxides

    if npca is not None:
        print(f"    pca components: {npca}")

        valid_args["npca"] = npca

    if kcluster is not None:
        print(f"    k-means clusters: {kcluster}")

        valid_args["kcluster"] = kcluster

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

    if configdir is not None:
        print(f"    configdir: {configdir}")

        valid_args["configdir"] = configdir

    if perplexdir is not None:
        print(f"    perplexdir: {perplexdir}")

        valid_args["perplexdir"] = perplexdir

    if logfile is not None:
        print(f"    logfile: {logfile}")

        valid_args["logfile"] = logfile

    if dataset is not None:
        print(f"    dataset: {dataset}")

        valid_args["dataset"] = dataset

    if emsonly is not None:
        if not isinstance(emsonly, bool):
            raise ValueError(
                "Invalid --emsonly argument ...\n"
                "--emsonly must be either True or False"
            )

        print(f"    endmembers only: {emsonly}")

        valid_args["emsonly"] = emsonly

    if verbose is not None:
        if not isinstance(verbose, bool):
            raise ValueError(
                "Invalid --verbose argument ...\n"
                "--verbose must be either True or False"
            )

        print(f"    verbose: {verbose}")

        valid_args["verbose"] = verbose

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return valid_args

#######################################################
##         Psuedosection Modeling Functions          ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+                  Helper Functions                 ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# Helper function to copy and replace perplex config files
def replace_in_file(file_path, replacements):
    """
    Replace specific strings in a text file with new values.

    Args:
        file_path (str): The path to the text file to be modified.
        replacements (dict): A dictionary where keys are the strings to be
            replaced, and values are the new strings to replace them with.

    Returns:
        None

    Example:
        To replace placeholders in a text file named "config.txt" with new values:
        ```
        replacements = {
            "{placeholder1}": "new_value1",
            "{placeholder2}": "new_value2"
        }
        replace_in_file("config.txt", replacements)
        ```
    """
    with open(file_path, "r") as file:
        file_data = file.read()
        for key, value in replacements.items():
            file_data = file_data.replace(key, value)
    with open(file_path, "w") as file:
        file.write(file_data)

# Move files from MAGEMin output dir
def cleanup_ouput_dir(sample_id, dataset, res, out_dir="runs"):
    """
    Move files from the MAGEMin output directory to a new directory based on the run name.
    Also moves the input data file into the new directory and removes the output directory.

    Args:
        sample_id (str): The name of the run, used to create the new directory and rename
            the files.
        dataset (str): The dataset type, e.g., "train" or "test".
        res (int): Resolution used in the MAGEMin run.
        out_dir (str, optional): The directory where MAGEMin outputs are stored (default: "runs").

    Returns:
        None

    Example:
        To organize MAGEMin output files for a specific run:
        ```
        cleanup_output_dir(sample_id="sample001", dataset="test", res=100)
        ```
    """
    # Get current working dir for making absolute paths
    cwd = os.getcwd()

    # Create the directory based on the sample_id
    model_out_dir = f"{cwd}/{out_dir}/magemin_{sample_id}_{dataset}_{res}"
    os.makedirs(model_out_dir, exist_ok=True)

    # Get a list of all files in the "output" directory matching the pattern
    files = os.listdir("output")
    matching_files = [file for file in files if file.startswith("_pseudosection")]

    # Rename and move the files
    for file in matching_files:
        new_filename = f"magemin-{sample_id}-{dataset}-{res}{file[len('_pseudosection'):]}"
        new_filepath = os.path.join(model_out_dir, new_filename)
        old_filepath = os.path.join("output", file)

        # Copy the file to the new location
        shutil.copy(old_filepath, new_filepath)

        # Remove the old file
        if os.path.exists(old_filepath):
            os.remove(old_filepath)

    # Move input data file into directory
    input_data_file = f"{out_dir}/magemin-{sample_id}-{dataset}-{res}.dat"
    input_data_destination = f"{model_out_dir}/magemin-{sample_id}-{dataset}-{res}.dat"

    if os.path.isfile(input_data_file):
        # Copy the input data file to the new location
        shutil.copy(input_data_file, input_data_destination)

        # Remove the old input data file
        os.remove(input_data_file)

    # Remove MAGEMin output directory
    shutil.rmtree("output")

# Get comp time for MAGEMin and Perple_X models
def get_comp_time(log_file, sample_id, dataset, res, nprocs, data_dir="assets/data"):
    """
    Extracts computation time information from a log file and appends it to a CSV file.

    Args:
        log_file (str): The path to the log file generated during MAGEMin and Perple_X runs.
        sample_id (str): The name of the sample or run.
        dataset (str): The dataset type, e.g., "train" or "test".
        res (int): The resolution used in the run.
        nprocs (int): The number of processor cores used in the run.
        data_dir (str, optional): The directory where CSV data files are stored
            (default: "assets/data").

    Returns:
        None

    Reads the specified log file, extracts the computation time information for MAGEMin and
        Perple_X,
    and appends it to a CSV file in the specified data directory.

    The CSV file has the following columns: "Sample ID", "Software", "Grid Size",
        "Computation Time (s)".
    """
    # Get current date
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%d-%m-%Y")

    # Define the CSV filename
    csv_filename = f'benchmark-gfem-efficiency-{formatted_date}.csv'

    # Define the full path to the CSV file
    csv_filepath = f"{data_dir}/{csv_filename}"

    if os.path.exists(log_file) and os.path.exists(data_dir):
        # Define a list to store the time values
        time_values_mgm = []
        time_values_ppx = []

        # Open the log file and read its lines
        with open(log_file, 'r') as log_file:
            lines = log_file.readlines()

        # Iterate over the lines in reverse order
        for line in reversed(lines):
            # Look for the line containing "MAGEMin comp time:"
            if "MAGEMin comp time:" in line:
                match = re.search(r"\+([\d.]+) ms", line)
                if match:
                    time_ms = float(match.group(1))
                    if nprocs <= 0:
                        time_s = time_ms / 1000
                    else:
                        time_s = time_ms / 1000 * nprocs
                    time_values_mgm.append(time_s)
                # Break the loop after finding the first match
                break

        # Iterate over the lines in reverse order
        for line in reversed(lines):
            if "Total elapsed time" in line:
                match = re.search(r"\s+([\d.]+)", line)
                if match:
                    time_m = float(match.group(1))
                    time_s = time_m * 60
                    time_values_ppx.append(time_s)
                # Break the loop after finding the first match
                break

        if time_values_mgm:
            # Get the last time value (most recent)
            last_value_mgm = time_values_mgm[-1]

            # Create the line to append to the CSV file
            line_to_append = f"{sample_id},magemin,{res*res},{last_value_mgm:.1f}"

            # Append the line to the CSV file
            with open(csv_filepath, 'a') as csv_file:
                csv_file.write(line_to_append + '\n')

        if time_values_ppx:
            # Get the last time value (most recent)
            last_value_ppx = time_values_ppx[-1]

            # Create the line to append to the CSV file
            line_to_append = f"{sample_id},perplex,{res*res},{last_value_ppx:.1f}"

            # Append the line to the CSV file
            with open(csv_filepath, 'a') as csv_file:
                csv_file.write(line_to_append + '\n')

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+          Sampling Bulk Rock Compositions          ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Read earthchem data
def read_earthchem_data(
        oxides=["SIO2","AL2O3","CAO","MGO","FEO","K2O","NA2O","TIO2","FE2O3","CR2O3"],
        data_dir="assets/data/"):
    """
    Reads a CSV file containing geochemical data and returns the data as a pandas DataFrame.

    Parameters:
        oxides (list of str, optional): List of oxides to include in the analysis.
            Defaults to a list of common oxides.
        data_dir (str, optional): The directory where the data files are located.
            Defaults to "assets/data/".

    Returns:
        pandas.DataFrame: The geochemical data read from the CSV file.

    This function reads multiple CSV files, processes them, and combines them into a single DataFrame. It also applies filtering criteria to the data based on the specified oxides. The function then prints information about the filtering criteria and summary statistics of the combined and filtered dataset.
    """
    # Find earthchem data files
    datafiles = [
        file for file in os.listdir(data_dir) if file.startswith("earthchem-igneous")
    ]

    # Filter criteria
    metadata = ["SAMPLE ID", "LATITUDE", "LONGITUDE", "COMPOSITION"]

    # Read all datafiles into dataframes
    dataframes = {}
    df_name = []
    for file in datafiles:
        df_name.append(file.split("-")[-1].split(".")[0])
        idx = file.split("-")[-1].split(".")[0]
        dataframes[f"df_{idx}"] = pd.read_csv(f"{data_dir}/{file}", delimiter="\t")
        dataframes[f"df_{idx}"] = dataframes[f"df_{idx}"][metadata + oxides]

    data = pd.concat(dataframes, ignore_index=True)
    if "SIO2" in oxides:
        data = data[data["SIO2"] >= 25]
        data = data[data["SIO2"] <= 90]
    if "CAO" in oxides:
        data = data[data["CAO"] <= 25]
    if "FE2O3" in oxides:
        data = data[data["FE2O3"] <= 20]
    if "TIO2" in oxides:
        data = data[data["TIO2"] <= 10]

    # Print info
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("Eartchem search portal critera:")
    print("    material: bulk")
    print("    normalization: oxides as reported")
    print("    sample type:")
    for name in df_name:
        print(f"        igneos > {name}")
    print("    oxides: (and/or)")
    for oxide in oxides:
        print(f"        {oxide}")
    print("Dataset filtering:")
    if "SIO2" in oxides:
        print("    SIO2 >= 25 wt.%")
        print("    SIO2 <= 90 wt.%")
    if "CAO" in oxides:
        print("    CAO <= 25 wt.%")
    if "FE2O3" in oxides:
        print("    FE2O3 <= 20 wt.%")
    if "TIO2" in oxides:
        print("    TIO2 <= 10 wt.%")
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Combined and filtered samples summary:")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(data[oxides].describe().map("{:.4g}".format))

    return data

# Get sample composition for MAGEMin and Perple_X models
def get_sample_composition(datafile, sample_id):
    """
    Extracts the oxide compositions needed for the "MAGEMin" and "Perple_X" analyses
    for a single sample specified by name from the given datafile.

    Args:
        datafile (str): Path to the CSV data file.
        sample_id (str): Name of the sample to select.

    Returns:
        list: Oxide compositions for the specified sample.

    This function reads a CSV data file, extracts the oxide compositions for a specific sample specified by its name (sample_id), and returns them as a list. The oxide compositions are needed for both the "MAGEMin" and "Perple_X" analyses. If the sample name is not found in the dataset, a ValueError is raised.
    """
    # All oxides needed for MAGEMin
    oxides = [
        "SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O",
        "NA2O", "TIO2", "FE2O3", "CR2O3", "H2O"
    ]

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
            if oxide != "H2O":
                composition.append(0.01)
            else:
                composition.append(0.00)

    return composition

# Sample randomly from the earthchem database
def get_random_sample_compositions(datafile, n=1, seed=None):
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

    This function reads a CSV data file, randomly samples n rows from it, and extracts the oxide compositions needed for the "MAGEMin" analysis for each randomly selected sample. The sample_ids list contains the sample IDs, and the compositions list contains lists of oxide compositions for each random sample. You can specify the number of samples to extract (n) and set a random seed for reproducibility (seed).
    """
    # All oxides needed for MAGEMin
    oxides = [
        "SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O",
        "NA2O", "TIO2", "FE2O3", "CR2O3", "H2O"
    ]

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

# Sample batches from the earthchem database
def get_batch_sample_compositions(datafile, batch_size=1, k=0):
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

    This function reads a CSV data file in batches of the specified size (batch_size) and returns the k-th batch. The sample_ids list contains the sample IDs, and the compositions list contains lists of oxide compositions for the samples in the k-th batch. You can specify the batch size and the index of the batch to retrieve (k).
    """
    # All oxides needed for MAGEMin
    oxides = [
        "SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O",
        "NA2O", "TIO2", "FE2O3", "CR2O3", "H2O"
    ]

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

    This function normalizes the concentrations of a subset of components within a sample. The input "sample" is a list of concentrations representing the components. You can specify the "components" parameter as a list of components to normalize. If "components" is set to "all," no normalization is performed, and the original sample is returned. The function returns a list of normalized concentrations for each component in the same order as the input.

    If the input sample list does not have exactly 11 components, a ValueError is raised.
    """
    # No normalizing for all components
    if components == "all":
        return sample

    # MAGEMin req components
    oxides = [
        "SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O",
        "NA2O", "TIO2", "FE2O3", "CR2O3", "H2O"
    ]

    # Check input
    if len(sample) != 11:
        error_message = (
            f"The input sample list must have exactly 11 components ...\n" +
            f"{oxides}"
        )
        raise ValueError(error_message)

    # Filter components
    subset_sample = [
        c if comp in components else 0.01 for c, comp in zip(sample, oxides)
    ]

    # Normalize
    total_subset_concentration = sum([c for c in subset_sample if c != 0.01])
    normalized_concentrations = []

    for c, comp in zip(sample, oxides):
        if comp in components:
            normalized_concentration = (
                (c / total_subset_concentration) * 100 if c != 0.01 else 0.01
            )
        else:
            normalized_concentration = 0.01
        normalized_concentrations.append(normalized_concentration)

    return normalized_concentrations

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+                 MAGEMin Functions                 ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Configure MAGEMin model
def configure_magemin(P_min, P_max, T_min, T_max, res, source,
                      sample_id, normox, dataset, out_dir="runs"):
    """
    Creates an input string for MAGEMin.

    Args:
        P_min (float): Minimum pressure (GPa) for the MAGEMin run.
        P_max (float): Maximum pressure (GPa) for the MAGEMin run.
        T_min (float): Minimum temperature (°C) for the MAGEMin run.
        T_max (float): Maximum temperature (°C) for the MAGEMin run.
        res (int): Resolution for pressure and temperature grids.
        source (str): Path to the CSV data file.
        sample_id (str): Name of the sample to use for the MAGEMin run.
        normox (list): List of oxide components for normalization.
        dataset (str): Dataset type ("train" or "test").
        out_dir (str, optional): Output directory for MAGEMin input files (default: "runs").

    Returns:
        None

    Prints:
        - Information about the created MAGEMin input file.
        - Pressure range and temperature range.
        - Composition [wt%] of the oxides.

    Creates:
        - A directory "runs" if it does not exist.
        - Writes the input string for MAGEMin to a file named "sample_id.dat"
        inside the "runs" directory.
    """
    # Create directory
    model_out_dir = f"{os.getcwd()}/{out_dir}"
    os.makedirs(model_out_dir, exist_ok=True)

    # Get sample composition
    sample_comp = get_sample_composition(source, sample_id)

    # Normalize composition
    norm_comp = normalize_sample(sample_comp, normox)

    # Transform units to kbar C
    P_min, P_max, T_min, T_max = P_min * 10, P_max * 10, T_min - 273, T_max - 273

    if dataset != "train":
        # Define small P T step to shift training dataset
        P_step, T_step = 1, 25

        # Print shift
        print("Shifting training dataset by a small amount:")
        print(f"P_min: {P_min/10} GPa --> P_min: {P_min/10 + P_step/10} GPa")
        print(f"P_max: {P_max/10} GPa --> P_max: {P_max/10 - P_step/10} GPa")
        print(f"T_min: {T_min+273} K --> T_min: {T_min+273 + T_step} K")
        print(f"T_max: {T_max+273} K --> T_max: {T_max+273 - T_step} K")

        # Shift PT range
        P_min, P_max = P_min + P_step, P_max - P_step
        T_min, T_max = T_min + T_step, T_max - T_step

    # PT range
    P_range, T_range = [P_min, P_max, (P_max-P_min)/res], [T_min, T_max, (T_max-T_min)/res]

    # Setup PT vectors
    magemin_input = ""
    pressure_values = np.arange(
        float(P_range[0]),
        float(P_range[1]) + float(P_range[2]),
        float(P_range[2])
    ).round(3)
    temperature_values = np.arange(
        float(T_range[0]),
        float(T_range[1]) + float(T_range[2]),
        float(T_range[2])
    ).round(3)

    # Expand PT vectors into grid
    combinations = list(itertools.product(pressure_values, temperature_values))
    for p, t in combinations:
        magemin_input += (
            f"0 {p} {t} {norm_comp[0]} {norm_comp[1]} {norm_comp[2]} "
            f"{norm_comp[3]} {norm_comp[4]} {norm_comp[5]} {norm_comp[6]} "
            f"{norm_comp[7]} {norm_comp[8]} {norm_comp[9]} {norm_comp[10]}\n"
        )

    # Write input file
    with open(f"{model_out_dir}/magemin-{sample_id}-{dataset}-{res}.dat", "w") as f:
        f.write(magemin_input)

# Run MAGEMin
def run_magemin(sample_id, res, dataset, emsonly=True, parallel=True,
                nprocs=os.cpu_count()-2, out_dir="runs", verbose=True):
    """
    Runs MAGEMin with the specified parameters.

    Args:
        sample_id (str, optional): The name of the MAGEMin run. Default is "test".
        res (int): Resolution used in the MAGEMin run.
        dataset (str): The dataset type, e.g., "train" or "test".
        emsonly (bool, optional): Determines whether to use endmembers-only mode in MAGEMin.
            Default is True.
        parallel (bool, optional): Determines whether to run MAGEMin in parallel.
            Default is True.
        nprocs (int, optional): The number of processes to use in parallel execution.
            Default is os.cpu_count()-2.
        out_dir (str, optional): The directory where MAGEMin outputs are stored (default: "runs").
        verbose (bool, optional): Determines whether to print verbose output.

    Returns:
        None

    Prints:
        - Information about the execution of MAGEMin, including the command executed
          and elapsed time.
        - Prints the elapsed time in seconds.
    """
    # Get current date
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%d-%m-%Y")

    # Log file
    log_file = f"log/log-magemin-{sample_id}-{dataset}-{res}-{formatted_date}"

    # Check for MAGEMin repo
    if os.path.exists("MAGEMin"):
        if emsonly:
            # Modified MAGEMin config file
            config = "assets/config/magemin-init-hp-endmembers"
            old_config = "MAGEMIN/src/initialize.h"
            if os.path.exists(config):
                # Replace MAGEMin config file with modified (endmembers only) file
                subprocess.run(f"cp {config} {old_config}", shell=True)

        # Compile MAGEMin
        if verbose:
            subprocess.run("(cd MAGEMin && make)", shell=True, text=True)
        else:
            with open(os.devnull, "w") as null:
                subprocess.run("(cd MAGEMin && make)", shell=True, stdout=null, stderr=null)

    else:
        # MAGEMin repo not found
        print("MAGEMin does not exist!")
        print("Clone MAGEMin from:")
        print("    https://github.com/ComputationalThermodynamics/MAGEMin.git")
        sys.exit()

    # Count number of pt points to model with MAGEMin
    input_path = f"{out_dir}/magemin-{sample_id}-{dataset}-{res}.dat"
    n_points = count_lines(input_path)

    # Check for input MAGEMin input files
    if not os.path.exists(input_path):
        sys.exit("No MAGEMin input files to run ...")

    # Execute MAGEMin in parallel with MPI
    if parallel == True:
        if nprocs > os.cpu_count():
            print(f"Number of processors {os.cpu_count()} is less than nprocs argument ...")
            print(f"Setting nprocs to {os.cpu_count() - 2}")
            nprocs = os.cpu_count() - 2
        elif nprocs < os.cpu_count():
            nprocs = nprocs
        exec = (
            f"mpirun -np {nprocs} MAGEMin/MAGEMin --File={input_path} "
            f"--n_points={n_points} --sys_in=wt --db=ig"
        )
    # Or execute MAGEMin in serial
    else:
        exec = (
            f"MAGEMin/MAGEMin --File={input_path} "
            f"--n_points={n_points} --sys_in=wt --db=ig"
        )

    # Run MAGEMin
    if verbose:
        subprocess.run(exec, shell=True, text=True)
    else:
        # Write to logfile
        with open(log_file, "w") as log:
            subprocess.run(exec, shell=True, stdout=log, stderr=log)

    # Move output files and cleanup directory
    cleanup_ouput_dir(sample_id, dataset, res, out_dir)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+                Perple_X Functions                 ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Configure Perple_X model
def configure_perplex(P_min, P_max, T_min, T_max, res, source, sample_id, normox,
                      dataset, emsonly=True, config_dir="assets/config",
                      perplex_dir="assets/perplex", out_dir="runs"):
    """
    Configures the Perple_X thermodynamic modeling software for a specific dataset and sample.

    Args:
        P_min (float): Minimum pressure in GPa for modeling.
        P_max (float): Maximum pressure in GPa for modeling.
        T_min (float): Minimum temperature in K for modeling.
        T_max (float): Maximum temperature in K for modeling.
        res (int): Resolution for modeling (e.g., number of data points).
        source (str): Path to the data source (e.g., CSV file) containing geochemical data.
        sample_id (str): Identifier for the sample being modeled.
        normox (list or str): List of components to normalize or "all" for no normalization.
        dataset (str): The dataset type, e.g., "train" or "test".
        emsonly (bool, optional): If True, use perplex-build-endmembers; if False,
            use perplex-build-solutions (default is True).
        config_dir (str, optional): Directory containing Perple_X configuration files
            (default is "assets/config").
        perplex_dir (str, optional): Directory containing Perple_X programs
            (default is "assets/perplex").
        out_dir (str, optional): Directory where Perple_X model outputs will be stored
            (default is "runs").

    Returns:
        None
    """
    # Get current working dir for making absolute paths
    cwd = os.getcwd()

    # Create directory
    abs_out_dir = f"{cwd}/{out_dir}"
    os.makedirs(abs_out_dir, exist_ok=True)

    # Get absolute path to perplex programs
    abs_perplex_dir = f"{cwd}/{perplex_dir}"

    # Create directory for storing perplex model outputs
    model_out_dir = f"{cwd}/{out_dir}/perplex_{sample_id}_{dataset}_{res}"

    # Remove old model
    if os.path.exists(model_out_dir):
        shutil.rmtree(model_out_dir)

    # Make new dir
    os.makedirs(model_out_dir, exist_ok=True)

    # Get sample composition
    sample_comp = get_sample_composition(source, sample_id)

    # Normalize composition
    norm_comp = normalize_sample(sample_comp, normox)

    # Transform units to kbar C
    P_min, P_max = P_min * 1e4, P_max * 1e4

    if dataset != "train":
        # Define small P T step to shift training dataset
        P_step, T_step = 1e3, 25

        # Print shift
        print("Shifting training dataset by a small amount:")
        print(f"P_min: {P_min/1e4} GPa --> P_min: {P_min/1e4 + P_step/1e4} GPa")
        print(f"P_max: {P_max/1e4} GPa --> P_max: {P_max/1e4 - P_step/1e4} GPa")
        print(f"T_min: {T_min} K --> T_min: {T_min + T_step} K")
        print(f"T_max: {T_max} K --> T_max: {T_max - T_step} K")

        # Shift PT range
        P_min, P_max = P_min + P_step, P_max - P_step
        T_min, T_max = T_min + T_step, T_max - T_step

    # Configuration files
    if emsonly:
        build = "perplex-build-endmembers"
    else:
        build = "perplex-build-solutions"
    min = "perplex-vertex-min"
    grid = "perplex-werami-grid"
    phase = "perplex-werami-phase"
    options = "perplex-build-options"
    draw = "perplex-pssect-draw"
    plot = "perplex-plot-options"

    # Copy original configuration files to the perplex directory
    shutil.copy(f"{config_dir}/{build}", f"{model_out_dir}/{build}")
    shutil.copy(f"{config_dir}/{min}", f"{model_out_dir}/{min}")
    shutil.copy(f"{config_dir}/{grid}", f"{model_out_dir}/{grid}")
    shutil.copy(f"{config_dir}/{phase}", f"{model_out_dir}/{phase}")
    shutil.copy(f"{config_dir}/{options}", f"{model_out_dir}/{options}")
    shutil.copy(f"{config_dir}/{draw}", f"{model_out_dir}/{draw}")
    shutil.copy(f"{config_dir}/{plot}", f"{model_out_dir}/perplex_plot_option.dat")

    # Modify the copied configuration files within the perplex directory
    replace_in_file(
        f"{model_out_dir}/{build}",
        {
            "{SAMPLEID}": f"{sample_id}-{dataset}-{res}",
            "{PERPLEX}": f"{abs_perplex_dir}",
            "{OUTDIR}": f"{model_out_dir}",
            "{TMIN}": str(T_min),
            "{TMAX}": str(T_max),
            "{PMIN}": str(P_min),
            "{PMAX}": str(P_max),
            "{SAMPLECOMP}": " ".join(map(str, norm_comp))
        }
    )
    replace_in_file(
        f"{model_out_dir}/{min}",
        {"{SAMPLEID}": f"{sample_id}-{dataset}-{res}"}
    )
    replace_in_file(
        f"{model_out_dir}/{grid}",
        {"{SAMPLEID}": f"{sample_id}-{dataset}-{res}"}
    )
    replace_in_file(
        f"{model_out_dir}/{phase}",
        {"{SAMPLEID}": f"{sample_id}-{dataset}-{res}"}
    )
    replace_in_file(
        f"{model_out_dir}/{options}",
        {
            "{XNODES}": f"{int(res / 4)} {res + 1}",
            "{YNODES}": f"{int(res / 4)} {res + 1}"
        }
    )
    replace_in_file(
        f"{model_out_dir}/{draw}",
        {"{SAMPLEID}": f"{sample_id}-{dataset}-{res}"}
    )

# Run Perple_X
def run_perplex(sample_id, dataset, res, emsonly=True, perplex_dir="assets/perplex",
                out_dir="runs", verbose=True):
    """
    Runs Perple_X with the specified parameters.

    Args:
        sample_id (str): The name of the Perple_X run.
        dataset (str): The dataset type, e.g., "train" or "test".
        res (int): The resolution used in the Perple_X run.
        emsonly (bool, optional): Determines whether to use endmembers-only mode in Perple_X.
            Default is True.
        perplex_dir (str, optional): The directory where Perple_X programs are located.
            Default is "assets/perplex".
        out_dir (str, optional): The directory where Perple_X outputs are stored (default: "runs").
        verbose (bool, optional): Determines whether to print verbose output.

    Returns:
        None

    Prints:
        - Information about the execution of Perple_X, including the command executed
          and elapsed time.
        - Prints the elapsed time in seconds.
    """
    # Get current working dir for making absolute paths
    cwd = os.getcwd()

    # Get absolute path to perplex programs
    abs_perplex_dir = f"{cwd}/{perplex_dir}"

    # Create directory for storing perplex model outputs
    model_out_dir = f"{cwd}/{out_dir}/perplex_{sample_id}_{dataset}_{res}"
    os.makedirs(model_out_dir, exist_ok=True)

    # Get current date
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%d-%m-%Y")

    # Log file
    log_file = f"log/log-perplex-{sample_id}-{dataset}-{res}-{formatted_date}"

    # Run programs with corresponding configuration files
    for program in ["build", "vertex", "werami", "pssect"]:
        # Get config files
        config_files = []

        if program == "build":
            if emsonly:
                config_files.append(f"{model_out_dir}/perplex-build-endmembers")
            else:
                config_files.append(f"{model_out_dir}/perplex-build-solutions")

        elif program == "vertex":
            config_files.append(f"{model_out_dir}/perplex-vertex-min")

        elif program == "werami":
            config_files.append(f"{model_out_dir}/perplex-werami-grid")
            config_files.append(f"{model_out_dir}/perplex-werami-phase")

        elif program == "pssect":
            config_files.append(f"{model_out_dir}/perplex-pssect-draw")

        # Get program path
        program_path = f"{cwd}/{perplex_dir}/{program}"

        for i, config in enumerate(config_files):
            try:
                # Set permissions
                os.chmod(program_path, 0o755)

                # Open the subprocess and redirect input from the input file
                with open(config, "rb") as input_stream:
                    process = subprocess.Popen(
                        [program_path],
                        stdin=input_stream,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        shell=True,
                        cwd=model_out_dir
                    )

                # Wait for the process to complete and capture its output
                stdout, stderr = process.communicate()

                # Write to logfile
                with open(log_file, "w") as log:
                    log.write(stdout.decode())
                    log.write(stderr.decode())

                if process.returncode != 0:
                    print(f"Error executing {program}: {stderr.decode()}")
                elif verbose:
                    print(f"{program} output:\n{stdout.decode()}")

                if program == "werami" and i == 0:
                    # Copy werami pseudosection output
                    shutil.copy(
                        f"{model_out_dir}/{sample_id}-{dataset}-{res}_1.tab",
                        f"{model_out_dir}/grid.tab"
                    )

                    # Remove old output
                    os.remove(f"{model_out_dir}/{sample_id}-{dataset}-{res}_1.tab")

                elif program == "werami" and i == 1:
                    # Copy werami mineral assemblage output
                    shutil.copy(
                        f"{model_out_dir}/{sample_id}-{dataset}-{res}_1.tab",
                        f"{model_out_dir}/phases.tab"
                    )

                    # Remove old output
                    os.remove(f"{model_out_dir}/{sample_id}-{dataset}-{res}_1.tab")

                elif program == "pssect":
                    # Copy pssect assemblages output
                    shutil.copy(
                        f"{model_out_dir}/{sample_id}-{dataset}-{res}_assemblages.txt",
                        f"{model_out_dir}/assemblages.txt"
                    )

                    # Copy pssect auto refine output
                    shutil.copy(
                        f"{model_out_dir}/{sample_id}-{dataset}-{res}_auto_refine.txt",
                        f"{model_out_dir}/auto_refine.txt"
                    )

                    # Copy pssect seismic data output
                    shutil.copy(
                        f"{model_out_dir}/{sample_id}-{dataset}-{res}_seismic_data.txt",
                        f"{model_out_dir}/seismic_data.txt"
                    )

                    # Remove old output
                    os.remove(f"{model_out_dir}/{sample_id}-{dataset}-{res}_assemblages.txt")
                    os.remove(f"{model_out_dir}/{sample_id}-{dataset}-{res}_auto_refine.txt")
                    os.remove(f"{model_out_dir}/{sample_id}-{dataset}-{res}_seismic_data.txt")

                    # Convert postscript file to pdf
                    ps = f"{model_out_dir}/{sample_id}-{dataset}-{res}.ps"
                    pdf = f"{model_out_dir}/{sample_id}-{dataset}-{res}.pdf"

                    subprocess.run(f"ps2pdf {ps} {pdf}", shell=True)

            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")

#######################################################
##                    ML Methods                     ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+                 Helper Functions                  ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Append info from a dict to csv
def append_to_csv(file_path, data_dict):
    """
    Append data from a dictionary to a CSV file.

    Parameters:
        file_path (str): The path to the CSV file where data will be appended or created.
        data_dict (dict): A dictionary containing the data to be appended to the CSV file.
                          Keys represent column names, and values are lists or arrays
                          containing the data to be appended.

    Returns:
        None

    This function checks if the specified CSV file exists. If it doesn't exist, a new
    DataFrame is created using the data from data_dict and saved as a new CSV file.
    If the CSV file already exists, the function loads the existing DataFrame,
    appends the new data from data_dict, and sorts the DataFrame by specified columns.
    The updated DataFrame is then saved back to the CSV file.
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

# Extract features array from training dataset
def extract_features(results):
    """
    Parameters:
        results (dict): A dictionary containing the training dataset results.

    Returns:
        tuple: A tuple containing the following:
            - P (list): Pressure values in MPa.
            - T (list): Temperature values in Kelvin.
            - features_array (numpy.ndarray): A 3D numpy array of shape (W, W, 2) representing
              the features array with pressure and temperature grids.

    Notes:
        - This function processes pressure (P) and temperature (T) values from the input
            results.
        - It transforms units and organizes the data into a 3D features array for training
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+   PCA and Synthetic Sampling Along Mixing Lines   ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Sample Earthchem data, run PCA, K-means clustering, and draw synthetic rock samples
# along mixing lines between cluster centroids
def earthchem_samples_pca(res, oxides, n_pca_components=3, k_pca_clusters=3, seed=42,
                          palette="tab10", figwidth=6.3, figheight=6.3, fontsize=22,
                          filename="earthchem-samples", fig_dir="figs",
                          data_dir="assets/data"):
    """
    Parameters:
        res (int): The resolution value.
        oxides (list): A list of oxide names to include in the analysis.
        n_pca_components (int, optional): The number of PCA components to consider.
            Default is 3.
        k_pca_clusters (int, optional): The number of PCA clusters to use. Default is 3.
        seed (int, optional): The random seed for reproducibility. Default is 42.
        palette (str, optional): The color palette to use for plotting. Default is "tab10".
        figwidth (float, optional): The width of the plot in inches. Default is 6.3.
        figheight (float, optional): The height of the plot in inches. Default is 6.3.
        fontsize (int, optional): The font size for text in the plot. Default is 22.
        filename (str, optional): The filename to save the plot. If not provided,
            the plot will be displayed interactively.
        fig_dir (str, optional): The directory to save the plot. Default is "./figs".
        data_dir (str, optional): The directory where data is stored. Default is
            "assets/data".

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

    # Ignore warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

    # SIO2 required to be in oxides list
    if "SIO2" not in oxides:
        oxides = ["SIO2"] + oxides

    # Read geochemical data
    print("Reading Earthchem samples ...")
    data = read_earthchem_data(oxides=oxides)

    # Sort by composition
    if "MGO" in oxides:
        data.sort_values(
            by=["SIO2", "MGO"],
            ascending=[True, False],
            inplace=True,
            ignore_index=True
        )
    else:
        data.sort_values(
            by="SIO2",
            ascending=True,
            inplace=True,
            ignore_index=True
        )

    # Impute missing measurement values by K-nearest algorithm
    imputer = KNNImputer(n_neighbors=4, weights="distance")
    imputer.fit(data[oxides])

    # Add missing values back to data
    data[oxides] = imputer.transform(data[oxides])

    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[oxides])

    # PCA modeling
    pca = PCA(n_components=n_pca_components)
    pca.fit(data_scaled)

    # Print summary
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("PCA summary for Earthchem samples:")
    print(f"    number of samples: {pca.n_samples_}")
    print(f"    PCA components: {n_pca_components}")
    print(f"    K-means clusters: {k_pca_clusters}")
    print(f"    features ({len(oxides)}):")
    for oxide in oxides:
        print(f"        {oxide}")
    print("    explained variance:")
    for i, value in enumerate(pca.explained_variance_ratio_):
        print(f"        PC{i+1}: {round(value, 3)}")
    print("    cumulative explained variance:")
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    for i, value in enumerate(cumulative_variance):
        print(f"        PC{i+1}: {round(value, 3)}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Transform the data to obtain the principal components
    principal_components = pca.transform(data_scaled)

    # Create a DataFrame to store the results
    pca_columns = [f"PC{i+1}" for i in range(n_pca_components)]
    data[pca_columns] = principal_components

    # Round numerical data
    data[oxides + pca_columns] = data[oxides + pca_columns].round(3)

    # Write csv file
    data.to_csv(f"{data_dir}/earthchem-samples-pca.csv", index=False)

    # Plot PCA loadings
    loadings = pd.DataFrame(
        (pca.components_.T * np.sqrt(pca.explained_variance_)).T,
        columns=oxides
    )

    # Colormap
    colormap = plt.cm.get_cmap(palette)

    # Plot PCA loadings
    fig = plt.figure(figsize=(figheight, figwidth))
    for i in [0, 1]:
        ax = fig.add_subplot(2, 1, i+1)
        ax.bar(oxides, loadings.iloc[i], color=colormap(i))
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("")
        ax.set_ylim([-1, 1])
        ax.set_ylabel("")
        ax.xaxis.set_major_locator(FixedLocator(range(len(oxides))))
        ax.set_xticklabels(oxides, rotation=90)
        plt.title(f"PC{i+1} Loadings")
        if i == 0:
            ax.set_xticks([])

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}-pca-loadings.png")
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

    # Plot PCA results
    for n in range(n_pca_components-1):
        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        for i, comp in enumerate(["ultramafic", "mafic", "intermediate", "felsic"]):
            indices = data.loc[data["COMPOSITION"] == comp].index
            scatter = ax.scatter(
                data.loc[indices, f"PC{n+1}"],
                data.loc[indices, f"PC{n+2}"],
                edgecolors="none",
                color=colormap(i),
                marker=".",
                label=comp
            )
        for oxide in oxides:
            ax.arrow(
                0, 0,
                loadings.at[n, oxide] * 3,
                loadings.at[n+1, oxide] * 3,
                width=0.02,
                head_width=0.14,
                color="black"
            )
            ax.text(
                (loadings.at[n, oxide] * 3) + (loadings.at[n, oxide] * 1),
                (loadings.at[n+1, oxide] * 3) + (loadings.at[n+1, oxide] * 1),
                oxide,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, pad=0.1),
                fontsize=fontsize * 0.579,
                color="black",
                ha = "center",
                va = "center"
            )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=4,
            columnspacing=0,
            markerscale=3,
            handletextpad=-0.5,
            fontsize=fontsize * 0.694
        )
        ax.set_xlabel(f"PC{n+1}")
        ax.set_ylabel(f"PC{n+2}")
        plt.title("Earthchem Samples")

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-pca{n+1}{n+2}.png")
        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    # Kmeans clustering in PCA space
    kmeans = KMeans(n_clusters=k_pca_clusters, n_init="auto", random_state=seed)
    kmeans.fit(principal_components)

    # Add cluster labels to data
    data["CLUSTER"] = kmeans.labels_

    # Get centroids
    centroids = kmeans.cluster_centers_
    original_centroids = pca.inverse_transform(centroids)

    # Plot PCA results and extract mixing lines among cluster centroids
    for n in range(n_pca_components-1):
        fig = plt.figure(figsize=(figwidth, figheight))
        ax = fig.add_subplot(111)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        for c in range(k_pca_clusters):
            # Get datapoints indices for each cluster
            indices = data.loc[data["CLUSTER"] == c].index
            scatter = ax.scatter(
                data.loc[indices, f"PC{n+1}"],
                data.loc[indices, f"PC{n+2}"],
                edgecolors="none",
                color=colormap(c+4),
                marker=".",
                alpha=0.3
            )
            clusters = ax.scatter(
                centroids[c, n],
                centroids[c, n+1],
                edgecolor="black",
                color=colormap(c+4),
                label=f"cluster {c+1}",
                marker="s",
                s=100
            )
            # Calculate mixing lines between cluster centroids
            if k_pca_clusters > 1:
                for i in range(c+1, k_pca_clusters):
                    m = ((centroids[i, n+1] - centroids[c, n+1]) /
                         (centroids[i, n] - centroids[c, n]))
                    b = centroids[c, n+1] - m * centroids[c, n]
                    x_vals = np.linspace(centroids[c, n], centroids[i, n], res)
                    y_vals = m * x_vals + b
                    ax.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=1)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=4,
            columnspacing=0,
            handletextpad=-0.5,
            fontsize=fontsize * 0.694
        )
        ax.set_xlabel(f"PC{n+1}")
        ax.set_ylabel(f"PC{n+2}")
        plt.title("Earthchem Samples")

        # Save the plot to a file if a filename is provided
        if filename:
            plt.savefig(f"{fig_dir}/{filename}-clusters{n+1}{n+2}.png")
        else:
            # Print plot
            plt.show()

        # Close device
        plt.close()

    # Initialize a dictionary for mixing lines
    mixing_lines = {}

    # Loop through PCA components
    print("Calculating mixing lines between cluster centroids:")
    for n in range(n_pca_components):
        for c in range(k_pca_clusters):
            # Calculate mixing lines between cluster centroids
            if k_pca_clusters > 1:
                for i in range(c+1, k_pca_clusters):
                    print(f"    PC{n+1}", f"cluster{c+1}", f"cluster{i+1}")
                    if n == 0:
                        mixing_lines[f"cluster{c+1}{i+1}"] = (
                            np.linspace(centroids[c, n], centroids[i, n], res)
                        )
                    else:
                        mixing_lines[f"cluster{c+1}{i+1}"] = np.vstack((
                            mixing_lines[f"cluster{c+1}{i+1}"],
                            np.linspace(centroids[c, n], centroids[i, n], res)
                        ))

    # Write mixing lines to csv
    print(f"Saving mixing lines to {data_dir} ...")
    for i in range(k_pca_clusters):
        for j in range(i+1, k_pca_clusters):
            data_synthetic = pd.DataFrame(
                np.hstack((
                    scaler.inverse_transform(
                        pca.inverse_transform(
                            mixing_lines[f"cluster{i+1}{j+1}"].T
                        )
                    ),
                    mixing_lines[f"cluster{i+1}{j+1}"].T
                )),
                columns=oxides + [f"PC{n+1}" for n in range(n_pca_components)]
            ).round(3)

            # Add sample id column
            data_synthetic.insert(
                0, "NAME",
                [f"c{i+1}{j+1}-{n}" for n in range(len(data_synthetic))]
            )

            # Write to csv
            data_synthetic.to_csv(
                f"{data_dir}/synthetic-samples-pca{n_pca_components}-clusters{i+1}{j+1}.csv",
                index=False
            )

    # Compile all synthetic datasets into a dict
    synthetic_datasets = {}
    for i in range(k_pca_clusters):
        for j in range(i+1, k_pca_clusters):
            synthetic_datasets[f"data_synthetic{i+1}{j+1}"] = pd.read_csv(
                f"{data_dir}/synthetic-samples-pca{n_pca_components}-clusters{i+1}{j+1}.csv"
            )

    # Create a grid of subplots
    num_plots = len(oxides) - 1
    if num_plots == 1:
        num_cols = 1
    elif num_plots > 1 and num_plots <= 4:
        num_cols = 2
    elif num_plots > 4 and num_plots <= 9:
        num_cols = 3
    elif num_plots > 9 and num_plots <= 16:
        num_cols = 4
    else:
        num_cols = 5
    num_rows = (num_plots + 1) // num_cols

    # Total figure size
    fig_width = figwidth / 2 * num_cols
    fig_height = figheight / 2 * num_rows

    # Harker diagrams
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    for k, y in enumerate([oxide for oxide in oxides if oxide != "SIO2"]):
        ax = axes[k]
        for i in range(k_pca_clusters):
            for j in range(i+1, k_pca_clusters):
                first_element = synthetic_datasets[f"data_synthetic{i+1}{j+1}"].iloc[0]
                last_element = synthetic_datasets[f"data_synthetic{i+1}{j+1}"].iloc[-1]
                sns.scatterplot(
                    data=synthetic_datasets[f"data_synthetic{i+1}{j+1}"],
                    x="SIO2",
                    y=y,
                    linewidth=0,
                    s=10,
                    zorder=100,
                    color=".2",
                    legend=False,
                    ax=ax
                )
                ax.annotate(
                    f"{i+1}",
                    xy=(first_element["SIO2"], first_element[y]),
                    xytext=(0, 0),
                    textcoords='offset points',
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        edgecolor="black",
                        facecolor="white",
                        alpha=0.6
                    ),
                    fontsize=fontsize * 0.579,
                    zorder=200,
                )
                ax.annotate(
                    f"{j+1}",
                    xy=(last_element["SIO2"], last_element[y]),
                    xytext=(0, 0),
                    textcoords='offset points',
                    bbox=dict(
                        boxstyle="round,pad=0.1",
                        edgecolor="black",
                        facecolor="white",
                        alpha=0.6
                    ),
                    fontsize=fontsize * 0.579,
                    zorder=200,
                )
        sns.kdeplot(
            data=data,
            x="SIO2",
            y=y,
            hue="COMPOSITION",
            hue_order=["ultramafic", "mafic", "intermediate", "felsic"],
            fill=False,
            ax=ax,
            levels=5
        )
        sns.scatterplot(
            data=data,
            x="SIO2",
            y=y,
            hue="COMPOSITION",
            hue_order=["ultramafic", "mafic", "intermediate", "felsic"],
            linewidth=0,
            s=5,
            zorder=1,
            legend=False,
            ax=ax
        )
        ax.set_title(f"{y}")
        ax.set_ylabel("")
        ax.set_xlabel("")
        if k < (num_plots - num_cols):
            ax.set_xticks([])
        if k == (num_plots - 1):
            handles = ax.get_legend().legendHandles
            labels = ["ultramafic", "mafic", "intermediate", "felsic"]
        for line in ax.get_legend().get_lines():
            line.set_linewidth(5)
        ax.get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4
    )
    fig.suptitle("Harker Diagrams vs. SIO2 (wt.%)")
    if num_plots < len(axes):
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}-harker-diagram.png")
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+      ML Training and K-fold Cross-Validation      ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
def cv_rocml(features_array, targets_array, features_array_valid, targets_array_valid,
             parameters, units, program, sample_id, model="DT", tune=False, seed=42,
             kfolds=10, parallel=True, nprocs=cpu_count()-2, vmin=None, vmax=None,
             palette="bone", figwidth=6.3, figheight=4.725, fontsize=22, filename=None,
             fig_dir="figs"):
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

# Train RocMLs
def train_rocml(sample_id, res, parameters, mask_geotherm=False, magemin=True,
                perplex=True, model="DT", tune=False, kfolds=cpu_count()-2, parallel=True,
                nprocs=cpu_count()-2, seed=42, palette="bone", out_dir="runs",
                fig_dir="figs", data_dir="assets/data"):
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
        results_mgm = process_magemin_results(sample_id, "train", res, out_dir)
        results_mgm_valid = process_magemin_results(sample_id, "valid", res, out_dir)

        # Get features arrays
        P_mgm, T_mgm, features_mgm = extract_features(results_mgm)
        P_mgm_valid, T_mgm_valid, features_mgm_valid = extract_features(results_mgm_valid)

    if perplex:
        # Get results
        results_ppx = process_perplex_results(
            f"{out_dir}/perplex_{sample_id}_train_{res}/grid.tab",
            f"{out_dir}/perplex_{sample_id}_train_{res}/assemblages.txt"
        )
        results_ppx_valid = process_perplex_results(
            f"{out_dir}/perplex_{sample_id}_valid_{res}/grid.tab",
            f"{out_dir}/perplex_{sample_id}_valid_{res}/assemblages.txt"
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

#######################################################
##                  Visualizations                   ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+   Post-processing MAGEMin and Perple_X results    ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
def process_magemin_results(sample_id, dataset, res, out_dir="runs"):
    """
    Process multiple MAGEMin output files in a directory based on a filename pattern.

    Args:
        sample_id (str): The name of the MAGEMin run.
        dataset (str): The name of the dataset (e.g., "train", "test").
        res (int): The resolution value.
        out_dir (str, optional): The directory where MAGEMin output files are stored.
            Default is "runs".

    Returns:
        dict: A single dictionary with merged key-value pairs from all files.

    Raises:
        FileNotFoundError: If the "runs" directory or the specific MAGEMin run
        directory does not exist.
    """
    # Check for MAGEMin output files
    if not os.path.exists(f"{out_dir}"):
        sys.exit("No MAGEMin output files to process ...")
    if not os.path.exists(f"{out_dir}/magemin_{sample_id}_{dataset}_{res}"):
        sys.exit("No MAGEMin output files to process ...")

    # Get filenames directory for files
    directory = f"{out_dir}/magemin_{sample_id}_{dataset}_{res}"
    pattern = f"magemin-{sample_id}-{dataset}-{res}_*.txt"

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

# Get Perple_X model results
def process_perplex_results(file_path_grid, file_path_assemblages):
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+             Other Helper Functions                ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Extract info along a geotherm
def extract_info_geotherm(results, parameter, thermal_gradient=0.5,
                          mantle_potential_T=1573, threshold=0.1):
    """
    Extracts relevant data points along a geotherm based on specified parameters.

    Parameters:
        results (dict): A dictionary containing geothermal data with keys "P" for pressure,
                        "T" for temperature, and a key for the specified parameter.
        parameter (str): The name of the parameter for data extraction along the geotherm.
        thermal_gradient (float, optional): The thermal gradient for geotherm calculation.
                                            Default is 0.5.
        mantle_potential_T (float, optional): The mantle potential temperature in Kelvin
                                              for geotherm calculation. Default is 1573 K.
        threshold (float, optional): The threshold for determining data points' proximity
                                     to the geotherm. Data points within this threshold
                                     of the calculated geotherm_P will be extracted.
                                     Default is 0.1.

    Returns:
        tuple: A tuple containing three arrays: P_values, T_values, and parameter_values.
               - P_values (numpy.ndarray): An array of pressure values along the geotherm.
               - T_values (numpy.ndarray): An array of temperature values along the geotherm.
               - parameter_values (numpy.ndarray): An array of the specified parameter's
                 values along the geotherm.

    This function takes geothermal data, calculates the geotherm, and extracts relevant data
    points corresponding to the specified parameter along the geotherm. It performs unit
    conversions for pressure and temperature as necessary before calculations.
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

# Crop geotherms within PT bounds
def crop_geotherms(P_array, T_array):
    """
    Crop geotherms within PT bounds.

    This function takes two arrays representing pressure and temperature.

    It creates four geotherm arrays (geotherm1, geotherm2, and geotherm3) by linearly
    spacing temperature values between 273 K and T_max. These geotherm arrays are
    calculated using different temperature gradients.

    The function proceeds to crop each geotherm array based on the PT bounds. Each geotherm
    is cropped to retain only the values that fall within the minimum and maximum pressure
    (P_min and P_max) and temperature (T_min and T_max) bounds obtained earlier.

    The resulting cropped geotherms and their corresponding temperature values are stored
    in separate lists, cropped_geotherms, and cropped_T_values.

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
        caption1 (str): Caption for the first image.
        caption2 (str): Caption for the second image.
        font_size (int, optional): Font size of the captions. Default is 150.
        caption_margin (int, optional): Margin between the captions and the images.
                                        Default is 25.
        dpi (int, optional): DPI (dots per inch) of the output image. Default is 330.
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
    Combine two plots vertically and add captions.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        output_path (str): Path to save the combined image with captions.
        caption1 (str): Caption for the first image.
        caption2 (str): Caption for the second image.
        font_size (int, optional): Font size of the captions. Default is 150.
        caption_margin (int, optional): Margin between the captions and the images.
                                        Default is 25.
        dpi (int, optional): DPI (dots per inch) of the output image. Default is 330.
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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+               Plotting Functions                  ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Plot pseudosection
def visualize_2d_pt_grid(P, T, grid, parameter, geotherm=False, geotherm_linetype="-.",
                         geotherm_color="white", T_unit="K", P_unit="GPa", title=None,
                         palette="blues", color_discrete=False, color_reverse=False,
                         vmin=None, vmax=None, figwidth=6.3, figheight=4.725, fontsize=22,
                         filename=None, fig_dir="figs"):
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

# Visualize benchmark comp times
def visualize_benchmark_gfem_times(datafile, palette="tab10", fontsize=12, figwidth=6.3,
                                   figheight=3.54, filename="benchmark-gfem-efficiency.png",
                                   fig_dir="figs"):
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
        - The function extracts x and y values from the filtered data for both mgm and ppx
            columns.
        - The function plots the data points and connects them with lines.
        - The function sets labels, title, and x-axis tick values.
        - The plot can be saved to a file if a filename is provided.
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read data
    data = pd.read_csv(datafile)

    # Arrange data by dataset resolution and sample
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
        bbox_to_anchor=(1.02, 0.5),
        loc="center left"
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
def visualize_training_PT_range(P_unit="GPa", T_unit="K", palette="tab10",
                                fontsize=12, figwidth=6.3, figheight=3.54,
                                filename="training-dataset-design.png", fig_dir="figs"):
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
        bbox_to_anchor=(1.02, 0.5)
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
def visualize_PREM(datafile, parameter, param_unit, results_mgm=None, results_ppx=None,
                   results_rocml=None, model=None, geotherm_threshold=0.1, P_unit="GPa",
                   depth=True, metrics=None, palette="tab10", title=None, figwidth=6.3,
                   figheight=4.725, fontsize=22, filename=None, fig_dir="figs"):
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
def visualize_GFEM(program, sample_id, res, dataset, parameters, mask_geotherm=True,
                   palette="bone", out_dir="runs", fig_dir="figs", data_dir="assets/data"):
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
        results = process_magemin_results(sample_id, dataset, res, out_dir)
    elif program == "Perple_X":
        # Get perplex results
        results = process_perplex_results(
            f"{out_dir}/perplex_{sample_id}_{dataset}_{res}/grid.tab",
            f"{out_dir}/perplex_{sample_id}_{dataset}_{res}/assemblages.txt"
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
def visualize_GFEM_diff(sample_id, res, dataset, parameters, mask_geotherm=True,
                        palette="bone", out_dir="runs", fig_dir="figs",
                        data_dir="assets/data"):
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
    results_mgm = process_magemin_results(sample_id, dataset, res, out_dir)

    # Get perplex results
    results_ppx = process_perplex_results(
        f"{out_dir}/perplex_{sample_id}_{dataset}_{res}/grid.tab",
        f"{out_dir}/perplex_{sample_id}_{dataset}_{res}/assemblages.txt"
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

# Visualize rocml performance
def visualize_rocml_performance(datafile="assets/data/benchmark-rocmls-performance.csv",
                                benchmark_times="assets/data/benchmark-gfem-efficiency.csv",
                                sample_id="PUM", parameter="DensityOfFullAssemblage",
                                res=128, palette="tab10", fontsize=22, figwidth=6.3,
                                figheight=4.725, filename="rocml", fig_dir="figs"):
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
            legend.set_bbox_to_anchor((0.48, 0.94))
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