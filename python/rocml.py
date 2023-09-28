#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
import pkg_resources
from git import Repo
import urllib.request
from tqdm import tqdm
from scipy import stats

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parallel computing !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing as mp
mp.set_start_method("fork")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import ListedColormap

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

#######################################################
## .1.  General Helper Functions for Scripting   !!! ##
#######################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read makefile variable !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_makefile_variable(makefile, variable):
    """
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read conda packages !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_conda_packages(condafile):
    """
    """
    try:
        with open(condafile, "r") as file:
            conda_data = yaml.safe_load(file)

        return conda_data.get("dependencies", [])

    except (IOError, yaml.YAMLError) as e:
        print(f"Error reading Conda file: {e}")

        return []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print session info !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def print_session_info(condafile=None, makefile=None):
    """
    """
    # Print session info
    print("Session info:")

    # Print Python version
    version_string = ".".join(map(str, sys.version_info))

    print(f"    Python Version: {version_string}")

    # Print package versions
    print("    Loaded packages:")

    if condafile:
        conda_packages = get_conda_packages(condafile)

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

    if makefile:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Makefile variables:")

        # Makefile assets
        makefile_vars_assets = ["DATADIR", "CONFIGDIR", "PERPLEXDIR"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_assets:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    Assets:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile dataset build options
        makefile_vars_dataset_build_options = [
            "SAMPLEID", "PMIN", "PMAX", "TMIN", "TMAX", "RES", "EMSONLY", "DATASET", "NORMOX",
            "SEED", "PARALLEL", "NPROCS", "KFOLDS", "VERBOSE"
        ]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_dataset_build_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    Dataset build options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile rocml options
        makefile_vars_rocml_options = ["TARGETS", "MLMODS", "MLTUNE"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_rocml_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    RocML options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile pca sampling options
        makefile_vars_pca_options = ["OXIDES", "NPCA", "KCLUSTER"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_pca_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    PCA options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        # Makefile visualization options
        makefile_vars_visualization_options = ["FIGDIR", "VISTARGETS", "COLORMAP"]

        # Get Makefile variables
        makefile_dict = {}

        for variable in makefile_vars_visualization_options:
            makefile_dict[f"{variable}"] = read_makefile_variable(makefile, variable)

        # Print variables
        print("    Visualization options:")
        for key, value in makefile_dict.items():
            print(f"        {key}: {value}")

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    else:
        print("No Makefile provided.")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# download and unzip !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def download_and_unzip(url, destination):
    """
    """
    # Download the file
    urllib.request.urlretrieve(url, "assets.zip")

    # Extract the contents of the zip file
    with zipfile.ZipFile("assets.zip", "r") as zip_ref:
        zip_ref.extractall(destination)

    # Remove the zip file
    os.remove("assets.zip")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# download github submodule !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def download_github_submodule(repository_url, submodule_dir, commit_hash):
    """
    """
    # Check if submodule directory already exists and delete it
    if os.path.exists(submodule_dir):
        shutil.rmtree(submodule_dir)

    # Clone submodule and recurse its contents
    try:
        repo = Repo.clone_from(repository_url, submodule_dir, recursive=True)
        repo.git.checkout(commit_hash)

    except Exception as e:
        print(f"An error occurred while cloning the GitHub repository: {e} ...")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compile magemin !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compile_magemin(emsonly, verbose):
    """
    """
    # Config dir
    config_dir = "assets/config"

    # Check for MAGEMin repo
    if os.path.exists("MAGEMin"):
        if emsonly:
            # Move modified MAGEMin config file with HP mantle endmembers
            config = f"{config_dir}/magemin-init-hp-endmembers"
            old_config = "MAGEMIN/src/initialize.h"

            if os.path.exists(config):
                # Replace MAGEMin config file with modified (endmembers only) file
                subprocess.run(f"cp {config} {old_config}", shell=True)

        # Compile MAGEMin
        if verbose >= 2:
            subprocess.run("(cd MAGEMin && make)", shell=True, text=True)

        else:
            with open(os.devnull, "w") as null:
                subprocess.run("(cd MAGEMin && make)", shell=True, stdout=null, stderr=null)

    else:
        # MAGEMin repo not found
        sys.exit("MAGEMin does not exist!")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# check non-matching strings !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_non_matching_strings(list1, list2):
    """
    """
    set1 = set(list1)
    set2 = set(list2)

    non_matching_strings = set1 - set2

    return bool(non_matching_strings)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parse list of strings !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_list_of_strings(arg):
    """
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parse arguments !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def parse_arguments():
    """
    """
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the command-line arguments
    parser.add_argument("--Pmin", type=int, required=False)
    parser.add_argument("--Pmax", type=int, required=False)
    parser.add_argument("--Tmin", type=int, required=False)
    parser.add_argument("--Tmax", type=int, required=False)
    parser.add_argument("--sampleid", type=str, required=False)
    parser.add_argument("--normox", type=parse_list_of_strings, required=False)
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--res", type=int, required=False)
    parser.add_argument("--benchmarks", type=str, required=False)
    parser.add_argument("--nsamples", type=int, required=False)
    parser.add_argument("--emsonly", type=str, required=False)
    parser.add_argument("--maskgeotherm", type=str, required=False)
    parser.add_argument("--targets", type=parse_list_of_strings, required=False)
    parser.add_argument("--models", type=parse_list_of_strings, required=False)
    parser.add_argument("--tune", type=str, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--batchp", type=float, required=False)
    parser.add_argument("--kfolds", type=int, required=False)
    parser.add_argument("--oxides", type=parse_list_of_strings, required=False)
    parser.add_argument("--npca", type=int, required=False)
    parser.add_argument("--kcluster", type=int, required=False)
    parser.add_argument("--parallel", type=str, required=False)
    parser.add_argument("--nprocs", type=int, required=False)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--palette", type=str, required=False)
    parser.add_argument("--figdir", type=str, required=False)
    parser.add_argument("--verbose", type=int, required=False)

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
    Pmin = args.Pmin
    Pmax = args.Pmax
    Tmin = args.Tmin
    Tmax = args.Tmax
    sampleid = args.sampleid
    normox = args.normox
    dataset = args.dataset
    res = args.res
    benchmarks = args.benchmarks
    nsamples = args.nsamples
    emsonly = args.emsonly
    maskgeotherm = args.maskgeotherm
    targets = args.targets
    models = args.models
    tune = args.tune
    epochs = args.epochs
    batchp = args.batchp
    kfolds = args.kfolds
    oxides = args.oxides
    npca = args.npca
    kcluster = args.kcluster
    parallel = args.parallel
    nprocs = args.nprocs
    seed = args.seed
    palette = args.palette
    figdir = args.figdir
    verbose = args.verbose

    # MAGEMin oxide options
    oxide_list_magemin = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O",
                          "TIO2", "FE2O3", "CR2O3", "H2O" ]

    valid_args = {}

    # Check arguments and print
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Running {script} with:")

    if Pmin is not None:
        print(f"    P min: {Pmin} GPa")

        valid_args["Pmin"] = Pmin

    if Pmax is not None:
        print(f"    P max: {Pmax} GPa")

        valid_args["Pmax"] = Pmax

    if Tmin is not None:
        print(f"    T min: {Tmin} K")

        valid_args["Tmin"] = Tmin

    if Tmax is not None:
        print(f"    T max: {Tmax} K")

        valid_args["Tmax"] = Tmax

    if sampleid is not None:
        print(f"    sample id: {sampleid}")

        valid_args["sampleid"] = sampleid

    if normox is not None:
        if normox != "all":
            if check_non_matching_strings(normox, oxide_list_magemin):
                print("Warning: invalid --normox argument!")
                print(f"    Can only normalize to oxides {oxide_list_magemin}")
                print("Using normox = 'all'")

                normox = "all"

            else:
                print(f"    Normalizing composition to: {oxide}")

        valid_args["normox"] = normox

    if dataset is not None:
        print(f"    dataset: {dataset}")

        valid_args["dataset"] = dataset

    if res is not None:
        if res > 128:
            print("Warning: invalid --res argument!")
            print("    --res must be <= 128")
            print("Using res = 128")

            res = 128

        print(f"    resolution: {res} pts")

        if Tmin is not None and Tmax is not None:
            print(f"    T range: [{Tmin}, {Tmax}, {res}] K")

        if Pmin is not None and Pmax is not None:
            print(f"    P range: [{Pmin}, {Pmax}, {res}] GPa")

        valid_args["res"] = res

    if benchmarks is not None:
        benchmarks = benchmarks.lower() == "true" if benchmarks else False

        if not isinstance(benchmarks, bool):
            print("Warning: invalid --benchmarks argument!")
            print("    --benchmarks must be True or False")
            print("Using benchmarks = True")

            benchmarks = True

        print(f"    benchmarks: {benchmarks}")

        valid_args["benchmarks"] = benchmarks

    if nsamples is not None:
        print(f"    n synthetic samples: {nsamples}")

        valid_args["nsamples"] = nsamples

    if emsonly is not None:
        emsonly = emsonly.lower() == "true" if emsonly else False

        if not isinstance(emsonly, bool):
            print("Warning: invalid --emsonly argument!")
            print("    --emsonly must be True or False")
            print("Using emsonly = False")

            emsonly = False

        print(f"    endmembers only: {emsonly}")

        valid_args["emsonly"] = emsonly

    if maskgeotherm is not None:
        maskgeotherm = maskgeotherm.lower() == "true" if maskgeotherm else False

        if not isinstance(maskgeotherm, bool):
            print("Warning: invalid --maskgeotherm argument!")
            print("    --maskgeotherm must be True or False")
            print("Using maskgeotherm = True")

            maskgeotherm = True

        print(f"    mask geotherm: {maskgeotherm}")

        valid_args["maskgeotherm"] = maskgeotherm

    if targets is not None:
        print(f"    targets: {targets}")

        valid_args["targets"] = targets

    if models is not None:
        print(f"    RocML models: {models}")

        valid_args["models"] = models

    if tune is not None:
        tune = tune.lower() == "true" if tune else False

        if not isinstance(tune, bool):
            print("Warning: invalid --tune argument!")
            print("    --tune must be True or False")
            print("Using tune = False")

            tune = False

        print(f"    hyperparameter tuning: {tune}")

        valid_args["tune"] = tune

    if epochs is not None:
        print(f"    NN epochs: {epochs}")

        valid_args["epochs"] = epochs

    if batchp is not None:
        print(f"    NN batch proportion: {batchp}")

        valid_args["batchp"] = batchp

    if kfolds is not None:
        print(f"    kfolds: {kfolds}")

        valid_args["kfolds"] = kfolds

    if oxides is not None:
        if check_non_matching_strings(oxides, oxide_list_magemin[:-1]):
            print("Warning: invalid --oxides argument!")
            print(f"    Can only use: {oxide_list_magemin[:-1]}")
            print(f"Using oxides = {oxide_list_magemin[:-1]}")

            oxides = oxide_list_magemin[:-1]

        print(f"    Selected oxides: {oxides}")

        valid_args["oxides"] = oxides

    if npca is not None:
        print(f"    pca components: {npca}")

        valid_args["npca"] = npca

    if kcluster is not None:
        print(f"    k-means clusters: {kcluster}")

        valid_args["kcluster"] = kcluster

    if parallel is not None:
        parallel = parallel.lower() == "true" if parallel else False

        if not isinstance(parallel, bool):
            print("Warning: invalid --parallel argument!")
            print("    --parallel must be True or False")
            print("Using parallel = True")

            parallel = True

        print(f"    parallel: {parallel}")

        valid_args["parallel"] = parallel

    if nprocs is not None:
        if nprocs > os.cpu_count():
            print(f"Warning: {nprocs} is greater than {os.cpu_count()} available processors!")
            print(f"Setting number of processors to: {os.cpu_count() - 2} ...")

            nprocs = os.cpu_count() - 2

        print(f"    processors: {nprocs}")

        valid_args["nprocs"] = nprocs

    if seed is not None:
        print(f"    seed: {seed}")

        valid_args["seed"] = seed

    if palette is not None:
        if palette not in ["viridis", "bone", "pink", "seismic", "grey", "blues"]:
            print(f"Warning: invalid --palette argument ({palette})!")
            print("    Palettes: viridis, bone, pink, seismic, grey, blues")
            print("Using palette = 'bone'")

            palette = "bone"

        print(f"    palette: {palette}")

        valid_args["palette"] = palette

    if figdir is not None:
        print(f"    fig directory: {figdir}")

        valid_args["figdir"] = figdir

    if verbose is not None:
        if not isinstance(verbose, int):
            print("Warning: invalid --verbose argument!")
            print("    --verbose must be an integer or 0")
            print("Using verbose = 1")

            verbose = 1

        print(f"    verbose: {verbose}")

        valid_args["verbose"] = verbose

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    return valid_args

#######################################################
## .2.          GFEM Modeling Functions          !!! ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .2.0            Helper Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# count lines !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def count_lines(filepath):
    """
    """
    line_count = 0

    with open(filepath, "r") as file:
        for line in file:
            line_count += 1

    return line_count

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# replace in file !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def replace_in_file(filepath, replacements):
    """
    """
    with open(filepath, "r") as file:
        file_data = file.read()

        for key, value in replacements.items():
            file_data = file_data.replace(key, value)

    with open(filepath, "w") as file:
        file.write(file_data)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get comp time !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_comp_time(program, sample_id, dataset, res):
    """
    """
    # Get current date
    formatted_date = datetime.datetime.now().strftime("%d-%m-%Y")

    # Define the CSV filename
    csv_filename = f"benchmark-efficiency.csv"

    # Data assets dir
    data_dir = "assets/data"

    # Define the full path to the CSV file
    csv_filepath = f"{data_dir}/{csv_filename}"

    # Log file
    log_file = f"log/log-{program}-{sample_id}-{dataset}-{res}-{formatted_date}"

    if os.path.exists(log_file) and os.path.exists(data_dir):
        # Define a list to store the time values
        time_values_mgm = []
        time_values_ppx = []

        # Open the log file and read its lines
        with open(log_file, "r") as log_file:
            lines = log_file.readlines()

        # Iterate over the lines in reverse order
        for line in reversed(lines):
            # Look for the line containing "MAGEMin comp time:"
            if "MAGEMin comp time:" in line:
                match = re.search(r"\+([\d.]+) ms", line)

                if match:
                    time_ms = float(match.group(1))
                    time_s = time_ms / 1000

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

            # Check if the CSV file already exists
            if not os.path.exists(csv_filepath):
                # If the file does not exist, write the header line first
                header_line = "sample,program,size,time"
                with open(csv_filepath, "w") as csv_file:
                    csv_file.write(header_line + "\n")

            # Append the line to the CSV file
            with open(csv_filepath, "a") as csv_file:
                csv_file.write(line_to_append + "\n")

        if time_values_ppx:
            # Get the last time value (most recent)
            last_value_ppx = time_values_ppx[-1]

            # Create the line to append to the CSV file
            line_to_append = f"{sample_id},perplex,{res*res},{last_value_ppx:.1f}"

            # Check if the CSV file already exists
            if not os.path.exists(csv_filepath):
                # If the file does not exist, write the header line first
                header_line = "sample,program,size,time"
                with open(csv_filepath, "w") as csv_file:
                    csv_file.write(header_line + "\n")

            # Append the line to the CSV file
            with open(csv_filepath, "a") as csv_file:
                csv_file.write(line_to_append + "\n")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .2.1     Sampling Bulk Rock Compositions      !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read earthchem data !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_earthchem_data(oxides, verbose):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Find earthchem data files
    filepaths = [
        file for file in os.listdir(data_dir) if file.startswith("earthchem-igneous")
    ]

    # Filter criteria
    metadata = ["SAMPLE ID", "LATITUDE", "LONGITUDE", "COMPOSITION"]

    # Read all filepaths into dataframes
    dataframes = {}
    df_name = []

    for file in filepaths:
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

    if verbose >= 2:
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get sample composition !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_sample_composition(filepath, sample_id):
    """
    """
    # All oxides needed for MAGEMin
    oxides = ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "FE2O3", "CR2O3",
              "H2O"]

    # Read the data file
    df = pd.read_csv(filepath)

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get random sampleids !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_random_sampleids(filepath, n=1, seed=None):
    """
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    # Select n random rows
    random_sampleids = df.sample(n)["NAME"]

    return random_sampleids.values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# normalize composition !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def normalize_composition(sample, components="all"):
    """
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
#+ .2.2            MAGEMin Functions             !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# configure magemin model !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def configure_magemin_model(P_min, P_max, T_min, T_max, res, source, sample_id, normox,
                            dataset):
    """
    """
    # Config dir
    config_dir = "assets/config"

    # Model output dir
    out_dir = "runs"

    # Model prefix
    model_prefix = f"{sample_id}-{dataset}-{res}"

    # Create directory for storing magemin model outputs
    model_out_dir = f"{out_dir}/magemin_{sample_id}_{dataset}_{res}"
    os.makedirs(model_out_dir, exist_ok=True)

    # Get sample composition
    sample_comp = get_sample_composition(source, sample_id)

    # Normalize composition
    norm_comp = normalize_composition(sample_comp, normox)

    # Transform units to kbar C
    P_min, P_max, T_min, T_max = P_min * 10, P_max * 10, T_min - 273, T_max - 273

    if dataset != "train":
        # Define small P T step to shift training dataset
        P_step, T_step = 1, 25

        # Shift PT range
        P_min, P_max = P_min + P_step, P_max - P_step
        T_min, T_max = T_min + T_step, T_max - T_step

    # PT range
    P_range, T_range = [P_min, P_max, (P_max-P_min)/res], [T_min, T_max, (T_max-T_min)/res]

    # Setup PT vectors
    magemin_input = ""

    P_array = np.arange(float(P_range[0]), float(P_range[1]) + float(P_range[2]),
                        float(P_range[2])).round(3)

    T_array = np.arange(float(T_range[0]), float(T_range[1]) + float(T_range[2]),
                        float(T_range[2])).round(3)

    # Expand PT vectors into grid
    combinations = list(itertools.product(P_array, T_array))

    for p, t in combinations:
        magemin_input += (
            f"0 {p} {t} {norm_comp[0]} {norm_comp[1]} {norm_comp[2]} "
            f"{norm_comp[3]} {norm_comp[4]} {norm_comp[5]} {norm_comp[6]} "
            f"{norm_comp[7]} {norm_comp[8]} {norm_comp[9]} {norm_comp[10]}\n"
        )

    # Write input file
    with open(f"{model_out_dir}/in.dat", "w") as f:
        f.write(magemin_input)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# iterate magemin sample !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def iterate_magemin_sample(args):
    """
    """
    # Set retries
    max_retries = 3

    for retry in range(max_retries):
        try:
            # Unpack arguments
            Pmin, Pmax, Tmin, Tmax, res, source, sampleid, normox, dataset, verbose = args

            # Configure Perple_X model
            configure_magemin_model(Pmin, Pmax, Tmin, Tmax, res, source, sampleid, normox,
                                    dataset)

            # Run Perple_X
            build_magemin_model(sampleid, dataset, res, verbose)

            # Get Perple_X comp time and write to csv
            get_comp_time("magemin", sampleid, dataset, res)

            # Process results
            process_magemin_results(sampleid, dataset, res, verbose)

            return None

        except Exception as e:
            print(f"Error occurred in attempt {retry + 1}: {e}")

            if retry < max_retries - 1:
                print(f"Retrying in 5 seconds ...")
                time.sleep(5)

            else:
                return e

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build magemin model !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_magemin_model(sample_id, dataset, res, verbose):
    """
    """
    # Model output dir
    out_dir = "runs"

    # Model prefix
    model_prefix = f"{sample_id}-{dataset}-{res}"

    # Create directory for storing magemin model outputs
    model_out_dir = f"{out_dir}/magemin_{sample_id}_{dataset}_{res}"
    os.makedirs(model_out_dir, exist_ok=True)

    # Model config file
    input_path = f"{model_out_dir}/in.dat"

    # Check for input MAGEMin input files
    if not os.path.exists(input_path):
        sys.exit("No MAGEMin input files to run!")

    # Get current date
    formatted_date = datetime.datetime.now().strftime("%d-%m-%Y")

    # Log file
    log_file = f"log/log-magemin-{model_prefix}-{formatted_date}"

    print(f"Building MAGEMin model: {sample_id} {dataset} {res}...")

    # Get number of pt points
    n_points = count_lines(input_path)

    # Execute MAGEMin
    exec = (f"../../MAGEMin/MAGEMin --File=../../{input_path} --n_points={n_points} "
            "--sys_in=wt --db=ig")

    try:
        # Run MAGEMin
        process = subprocess.Popen([exec], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, shell=True,
                                   cwd=f"{model_out_dir}")

        # Wait for the process to complete and capture its output
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error executing '{exec}':")

            if stderr is not None:
                print(f"{stderr.decode()}")
            else:
                print("No standard error output.")

        # Write to logfile
        with open(log_file, "a") as log:
            log.write(stdout.decode())

            if stderr is not None:
                log.write(stderr.decode())

        if verbose >= 2:
            print(f"MAGEMin output:")
            print(f"{stdout.decode()}")

    except subprocess.CalledProcessError as e:
        if verbose >= 2:
            print(f"Error: {e}")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .2.3           Perple_X Functions             !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# configure perplex model !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def configure_perplex_model(P_min, P_max, T_min, T_max, res, source, sample_id, normox,
                            dataset):
    """
    """
    # Get current working dir for making absolute paths
    cwd = os.getcwd()

    # Perple_X program dir
    perplex_dir = "assets/perplex"

    # Perple_X config dir
    config_dir = "assets/config"

    # Model output dir
    out_dir = "runs"

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

    # Model prefix
    model_prefix = f"{sample_id}-{dataset}-{res}"

    # Get sample composition
    sample_comp = get_sample_composition(source, sample_id)

    # Normalize composition
    norm_comp = normalize_composition(sample_comp, normox)

    # Transform units to kbar C
    P_min, P_max = P_min * 1e4, P_max * 1e4

    if dataset != "train":
        # Define small P T step to shift training dataset
        P_step, T_step = 1e3, 25

        # Shift PT range
        P_min, P_max = P_min + P_step, P_max - P_step
        T_min, T_max = T_min + T_step, T_max - T_step

    # Copy endmembers only or solid solution config
    shutil.copy(
        f"{config_dir}/perplex-build-solutions",
        f"{config_dir}/perplex-build-model"
    )

    # Configuration files
    build = "perplex-build-model"
    min = "perplex-vertex-min"
    targets = "perplex-werami-targets"
    phase = "perplex-werami-phase"
    options = "perplex-build-options"
    draw = "perplex-pssect-draw"
    plot = "perplex-plot-options"

    # Copy original configuration files to the perplex directory
    shutil.copy(f"{config_dir}/{build}", f"{model_out_dir}/{build}")
    shutil.copy(f"{config_dir}/{min}", f"{model_out_dir}/{min}")
    shutil.copy(f"{config_dir}/{targets}", f"{model_out_dir}/{targets}")
    shutil.copy(f"{config_dir}/{phase}", f"{model_out_dir}/{phase}")
    shutil.copy(f"{config_dir}/{options}", f"{model_out_dir}/{options}")
    shutil.copy(f"{config_dir}/{draw}", f"{model_out_dir}/{draw}")
    shutil.copy(f"{config_dir}/{plot}", f"{model_out_dir}/perplex_plot_option.dat")

    # Modify the copied configuration files within the perplex directory
    replace_in_file(f"{model_out_dir}/{build}",
                    {"{SAMPLEID}": f"{model_prefix}",
                     "{PERPLEX}": f"{abs_perplex_dir}",
                     "{OUTDIR}": f"{model_out_dir}",
                     "{TMIN}": str(T_min),
                     "{TMAX}": str(T_max),
                     "{PMIN}": str(P_min),
                     "{PMAX}": str(P_max),
                     "{SAMPLECOMP}": " ".join(map(str, norm_comp))})
    replace_in_file(f"{model_out_dir}/{min}", {"{SAMPLEID}": f"{model_prefix}"})
    replace_in_file(f"{model_out_dir}/{targets}", {"{SAMPLEID}": f"{model_prefix}"})
    replace_in_file(f"{model_out_dir}/{phase}", {"{SAMPLEID}": f"{model_prefix}"})
    replace_in_file(f"{model_out_dir}/{options}",
                    {"{XNODES}": f"{int(res / 4)} {res + 1}",
                     "{YNODES}": f"{int(res / 4)} {res + 1}"})
    replace_in_file(f"{model_out_dir}/{draw}", {"{SAMPLEID}": f"{model_prefix}"})

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# iterate perplex sample !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def iterate_perplex_sample(args):
    """
    """
    # Set retries
    max_retries = 3

    for retry in range(max_retries):
        try:
            # Unpack arguments
            Pmin, Pmax, Tmin, Tmax, res, source, sampleid, normox, dataset, verbose = args

            # Configure Perple_X model
            configure_perplex_model(Pmin, Pmax, Tmin, Tmax, res, source, sampleid, normox,
                                    dataset)

            # Run Perple_X
            build_perplex_model(sampleid, dataset, res, verbose)

            # Get Perple_X comp time and write to csv
            get_comp_time("perplex", sampleid, dataset, res)

            # Process results
            process_perplex_results(sampleid, dataset, res, verbose)

            return None

        except Exception as e:
            print(f"Error occurred in attempt {retry + 1}: {e}")

            if retry < max_retries - 1:
                print(f"Retrying in 5 seconds ...")
                time.sleep(5)

            else:
                return e

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build perplex model !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_perplex_model(sample_id, dataset, res, verbose):
    """
    """
    # Get current working dir for making absolute paths
    cwd = os.getcwd()

    # Perple_X program path
    perplex_dir = "assets/perplex"

    # Model output dir
    out_dir = "runs"

    # Get absolute path to perplex programs
    abs_perplex_dir = f"{cwd}/{perplex_dir}"

    # Create directory for storing perplex model outputs
    model_out_dir = f"{cwd}/{out_dir}/perplex_{sample_id}_{dataset}_{res}"
    os.makedirs(model_out_dir, exist_ok=True)

    # Model prefix
    model_prefix = f"{sample_id}-{dataset}-{res}"

    # Get current date
    formatted_date = datetime.datetime.now().strftime("%d-%m-%Y")

    # Log file
    log_file = f"log/log-perplex-{model_prefix}-{formatted_date}"

    print(f"Building Perple_X model: {sample_id} {dataset} {res} ...")

    # Run programs with corresponding configuration files
    for program in ["build", "vertex", "werami", "pssect"]:
        # Get config files
        config_files = []

        if program == "build":
            config_files.append(f"{model_out_dir}/perplex-build-model")

        elif program == "vertex":
            config_files.append(f"{model_out_dir}/perplex-vertex-min")

        elif program == "werami":
            config_files.append(f"{model_out_dir}/perplex-werami-targets")
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
                    process = subprocess.Popen([program_path], stdin=input_stream,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               shell=True, cwd=model_out_dir)

                # Wait for the process to complete and capture its output
                stdout, stderr = process.communicate()

                # Write to logfile
                with open(log_file, "a") as log:
                    log.write(stdout.decode())
                    log.write(stderr.decode())

                if process.returncode != 0:
                    print(f"Error executing perplex program '{program}'!")

                elif verbose >= 2:
                    print(f"{program} output:")
                    print(f"{stdout.decode()}")

                if program == "werami" and i == 0:
                    # Copy werami pseudosection output
                    shutil.copy(
                        f"{model_out_dir}/{model_prefix}_1.tab",
                        f"{model_out_dir}/target-array.tab"
                    )

                    # Remove old output
                    os.remove(f"{model_out_dir}/{model_prefix}_1.tab")

                elif program == "werami" and i == 1:
                    # Copy werami mineral assemblage output
                    shutil.copy(
                        f"{model_out_dir}/{model_prefix}_1.tab",
                        f"{model_out_dir}/phases.tab"
                    )

                    # Remove old output
                    os.remove(f"{model_out_dir}/{model_prefix}_1.tab")

                elif program == "pssect":
                    # Copy pssect assemblages output
                    shutil.copy(
                        f"{model_out_dir}/{model_prefix}_assemblages.txt",
                        f"{model_out_dir}/assemblages.txt"
                    )

                    # Copy pssect auto refine output
                    shutil.copy(
                        f"{model_out_dir}/{model_prefix}_auto_refine.txt",
                        f"{model_out_dir}/auto_refine.txt"
                    )

                    # Copy pssect seismic data output
                    shutil.copy(
                        f"{model_out_dir}/{model_prefix}_seismic_data.txt",
                        f"{model_out_dir}/seismic_data.txt"
                    )

                    # Remove old output
                    os.remove(f"{model_out_dir}/{model_prefix}_assemblages.txt")
                    os.remove(f"{model_out_dir}/{model_prefix}_auto_refine.txt")
                    os.remove(f"{model_out_dir}/{model_prefix}_seismic_data.txt")

                    # Convert postscript file to pdf
                    ps = f"{model_out_dir}/{model_prefix}.ps"
                    pdf = f"{model_out_dir}/{model_prefix}.pdf"

                    subprocess.run(f"ps2pdf {ps} {pdf}", shell=True)

            except subprocess.CalledProcessError as e:
                if verbose >= 2:
                    print(f"Error: {e}")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .2.3       Build GFEM Models Functions        !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# build gfem models !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_gfem_models(program, Pmin, Pmax, Tmin, Tmax, res, source, sampleids, normox,
                      parallel, nprocs, verbose):
    """
    """
    # Datasets to iterate over
    datasets = ["train", "valid"]

    # Create combinations of samples and datasets
    combinations = list(itertools.product(sampleids, datasets))

    # Define number of processors
    if not parallel:
        nprocs = 1

    elif parallel:
        if nprocs is None or nprocs > os.cpu_count():
            nprocs = os.cpu_count() - 2

        else:
            nprocs = nprocs

    # Make sure nprocs is not greater than sample combinations
    if nprocs > len(combinations):
        nprocs = len(combinations)

    # Create list of args for mp pooling
    run_args = [
        (Pmin, Pmax, Tmin, Tmax, res, source, sampleid, normox, dataset, verbose
        ) for sampleid, dataset in combinations
    ]

    # Create a multiprocessing pool
    with mp.Pool(processes=nprocs) as pool:
        if program == "magemin":
            results = pool.map(iterate_magemin_sample, run_args)

        elif program == "perplex":
            results = pool.map(iterate_perplex_sample, run_args)

        else:
            sys.exit("Program must be 'magemin' or 'perplex'")

        # Wait for all processes
        pool.close()
        pool.join()

    # Check for errors in the results
    error_count = 0

    for result in results:
        if result is not None:
            if verbose >= 2:
                print("Error occurred with GFEM model:", result)

            error_count += 1

    if error_count > 0:
        print(f"Total errors: {error_count}")

    else:
        print("All GFEM models built successfully!")

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#######################################################
## .3.   Post-processing MAGEMin and Perple_X    !!! ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .3.0             Helper Functions             !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# encode assemblages !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def encode_assemblages(assemblages, filepath):
    """
    """
    unique_assemblages = {}
    encoded_assemblages = []

    # Encoding unique phase assemblages
    for assemblage in assemblages:
        assemblage_tuple = tuple(sorted(assemblage))

        if assemblage_tuple not in unique_assemblages:
            unique_assemblages[assemblage_tuple] = len(unique_assemblages) + 1

    # Save list of unique phase assemblages
    if filepath is not None:
        # Create dataframe
        df = pd.DataFrame(list(unique_assemblages.items()), columns=["assemblage", "index"])

        # Put spaces between phases
        df["assemblage"] = df["assemblage"].apply(" ".join)

        # Save to csv
        df.to_csv(filepath, index=False)

    # Encoding phase assemblage numbers
    for assemblage in assemblages:
        if assemblage == "":
            encoded_assemblages.append(np.nan)

        else:
            encoded_assemblage = unique_assemblages[tuple(sorted(assemblage))]
            encoded_assemblages.append(encoded_assemblage)

    return encoded_assemblages

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create geotherm mask !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_geotherm_mask(results, T_mantle1=273, T_mantle2=1773, grad_mantle1=1,
                         grad_mantle2=0.5):
    """
    """
    # Get PT values
    P, T = results["P"], results["T"]

    # Get max PT
    P_min, P_max, T_min, T_max = np.min(P), np.max(P), np.min(T), np.max(T)

    # Find geotherm boundaries
    T1_Pmax = (P_max * grad_mantle1 * 35) + T_mantle1
    P1_Tmin = (T_min - T_mantle1) / (grad_mantle1 * 35)
    T2_Pmin = (P_min * grad_mantle2 * 35) + T_mantle2
    T2_Pmax = (P_max * grad_mantle2 * 35) + T_mantle2

    # Iterate through PT array and set nan where PT is out of geotherm bounds
    PT_array = np.stack((P, T), axis=-1)

    for i in range(PT_array.shape[0]):
        p = PT_array[i, 0]
        t = PT_array[i, 1]

        # Calculate mantle geotherms
        geotherm1 = (t - T_mantle1) / (grad_mantle1 * 35)
        geotherm2 = (t - T_mantle2) / (grad_mantle2 * 35)

        # Set PT array to nan if outside of geotherm bounds
        if (
               ((t <= T1_Pmax) and (p >= geotherm1)) or
               ((t >= T2_Pmin) and (p <= geotherm2))
        ):
            PT_array[i] = [np.nan, np.nan]

    # Create nan mask
    mask = np.isnan(PT_array[:,0])

    return mask

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read gfem results !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_gfem_results(program, sample_id, dataset, res, verbose):
    """
    """
    # Model output dir
    out_dir = "runs"

    # Get directory with magemin model outputs
    model_out_dir = f"{out_dir}/{program}_{sample_id}_{dataset}_{res}"

    # Check for MAGEMin output files
    if not os.path.exists(out_dir):
        sys.exit("No results to read!")

    # Get filepaths for magemin output
    filepath = f"{model_out_dir}/results.csv"

    if not os.path.exists(filepath):
        sys.exit("No results to read!")

    if verbose >= 1:
        print(f"Reading results: {model_out_dir} ...")

    # Read results
    df = pd.read_csv(filepath)

    # Initialize empty dict
    results = {}

    # Convert to dict of np arrays
    for column in df.columns:
        results[column] = df[column].values

    return results

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# impute array with nans !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def impute_array_with_nans(array, n_neighbors=1):
    """
    """
    # Create a copy of the input array to avoid modifying the original array
    result_array = array.copy()

    # Iterate through each element of the array
    for i in range(len(result_array)):
        if np.isnan(result_array[i]):
            # Define the neighborhood indices
            neighbors = ([i - j for j in range(1, n_neighbors + 1)] +
                         [i + j for j in range(1, n_neighbors + 1)])

            # Check if any of the surrounding values are also nan
            is_nan_surrounded = any(
                np.isnan(result_array[x]) for x in neighbors if 0 <= x < len(result_array)
            )

            if is_nan_surrounded:
                # If surrounded by nans, set the nan to 0
                result_array[i] = 0

            else:
                # If surrounded by numerical values, impute the mean value of neighbors
                surrounding_values = [
                    result_array[x] for x in neighbors
                    if 0 <= x < len(result_array) and not np.isnan(result_array[x])
                ]

                if surrounding_values:
                    result_array[i] = np.mean(surrounding_values)

                else:
                    # If there are no surrounding numerical values, set to 0
                    result_array[i] = 0

    return result_array

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create feature array !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_feature_array(results, mask_geotherm):
    """
    """
    # Get P T arrays
    P, T = results["P"], results["T"]

    # Mask geotherm
    if mask_geotherm:
        # Get geotherm mask
        mask = create_geotherm_mask(results)

        # Apply mask to features
        P[mask] = np.nan
        T[mask] = np.nan

    # Stack PT arrays
    feature_array = np.stack((P, T), axis=-1)

    return feature_array

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# create target array !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def create_target_array(results, targets, mask_geotherm):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Initialize empty list for target arrays
    target_array_list = []

    # Rearrange results to match targets
    results_rearranged = {key: results[key] for key in targets}

    # Get target arrays
    for key, value in results_rearranged.items():
        if key in targets:
            target_array_list.append(impute_array_with_nans(value))

    # Stack target arrays
    target_array = np.stack(target_array_list, axis=-1)

    if mask_geotherm:
        # Get geotherm mask
        mask = create_geotherm_mask(results)

        # Apply mask to all target arrays
        for j in range(target_array.shape[1]):
            target_array[:, j][mask] = np.nan

    return target_array

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .3.1             Process MAGEMin              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read magemin output !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_magemin_output(filepath):
    """
    """
    # Open file
    with open(filepath, "r") as file:
        lines = file.readlines()

    # Skip the comment line
    lines = lines[1:]

    # Initialize results
    results = []

    # Read lines
    while lines:
        # Get line with PT, Gamma, etc.
        line = lines.pop(0)

        # Split line at whitespace and transform into floats
        a = list(map(float, line.split()))

        # Get results
        num_point = int(a[0]) # PT point
        status = int(a[1]) # solution status
        p = a[2] # P kbar
        t = a[3] # T celcius
        gibbs = a[4] # gibbs free energy J
        br_norm = a[5] # normalized mass residual
        gamma = a[6:15] # chemical potential of pure components (gibbs hyperplane)
        unknown = a[16] # unknown parameter following gamma in output ???
        vp = a[17] # P-wave velocity km/s
        vs = a[18] # S-wave velocity km/s
        entropy = a[19] # entropy J/K

        # Initialize empty lists for stable solutions and endmember info
        assemblage = []
        assemblage_mode = []
        assemblage_rho = []
        compositional_vars = []
        em_fractions = []
        em_list = []

        # Get line with stable solutions and endmember info
        line = lines.pop(0)

        # Read line
        while line.strip():
            # Split line at whitespace
            out = line.split()

            # Initialize empty list to store stable solutions info
            data = []

            # Store stable solutions info as floats or strings (if not numeric)
            for value in out:
                try:
                    data.append(float(value))

                except ValueError:
                    data.append(value)

            # Store stable solutions info
            assemblage.append(data[0]) # phase assemblage
            assemblage_mode.append(data[1]) # assemblage mode
            assemblage_rho.append(data[2]) # assemblage density

            # Initialize empty lists for endmember info
            comp_var = []
            em_frac = []
            em = []

            if len(data) > 4:
                n_xeos = int(data[3]) # number of compositional variables
                comp_var = data[4:4 + n_xeos] # compositional variables
                em = out[4 + n_xeos::2] # endmembers
                em_frac = data[5 + n_xeos::2] # endmember fractions

            # Store endmember info
            compositional_vars.append(comp_var)
            em_fractions.append(em_frac)
            em_list.append(em)

            line = lines.pop(0)

        # Get indices of melt
        ind_liq = [idx for idx, sol in enumerate(assemblage) if sol == "liq"]

        # Get melt fraction
        if ind_liq:
            liq = assemblage_mode[ind_liq[0]]

            if liq <= 0.01:
                liq = np.nan

        else:
            liq = np.nan

        # Compute average density of full assemblage
        rho_total = sum(mode * rho for mode, rho in zip(assemblage_mode, assemblage_rho))

        # Compute density of liq
        if liq > 0:
            rho_liq = assemblage_rho[ind_liq[0]]

            # Get indices of stable phases
            ind_sol = [idx for idx, sol in enumerate(assemblage) if sol != "liq"]

            # Compute average density of stable phases
            if ind_sol:
                rho_sol = sum(mode * rho for mode, rho in zip(
                    [assemblage_mode[idx] for idx in ind_sol],
                    [assemblage_rho[idx] for idx in ind_sol]
                )) / sum([assemblage_mode[idx] for idx in ind_sol])

            else:
                rho_sol = 0

            # Compute density of solid-melt mixture
            rho_mix = (liq * rho_liq + (1 - liq) * rho_sol)

        else:
            rho_liq = 0
            rho_sol = 0
            rho_mix = 0

        # Append results dictionary
        results.append({"point": num_point, # point
                        "T": t, # temperature celcius
                        "P": p, # pressure kbar
                        "rho": rho_total, # density of full assemblage kg/m3
                        "Vp": vp, # pressure wave velocity km/s
                        "Vs": vs, # shear wave velocity km/s
                        "melt_fraction": liq, # melt fraction
                        "assemblage": assemblage, # stable assemblage
                        })

    # Merge lists within dictionary
    combined_results = {key: [d[key] for d in results] for key in results[0]}

    return combined_results

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# process magemin results !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_magemin_results(sample_id, dataset, res, verbose):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Model output dir
    out_dir = "runs"

    # Get directory with magemin model outputs
    model_out_dir = f"{out_dir}/magemin_{sample_id}_{dataset}_{res}"

    # Check for MAGEMin output files
    if not os.path.exists(out_dir):
        sys.exit("No MAGEMin files to process!")

    # Get filepaths for magemin output
    filepath = f"{model_out_dir}/output/_pseudosection_output.txt"

    if not os.path.exists(filepath):
        sys.exit("No MAGEMin files to process!")

    if verbose >= 1:
        print(f"Reading MAGEMin output: {model_out_dir} ...")

    # Read results
    results = read_magemin_output(filepath)

    # Remove point index
    results.pop("point")

    # Compute assemblage variance (number of phases)
    assemblages = results.get("assemblage")

    assemblage_variance = []

    for assemblage in assemblages:
        unique_phases = set(assemblage)
        count = len(unique_phases)
        assemblage_variance.append(count)

    # Add assemblage variance to merged results
    results["assemblage_variance"] = assemblage_variance

    # Encode assemblage
    csv_filepath = f"{model_out_dir}/assemblages.csv"
    encoded_assemblages = encode_assemblages(assemblages, csv_filepath)

    # Replace assemblage with encoded assemblages
    results["assemblage"] = encoded_assemblages

    # Point results that can be converted to numpy arrays
    point_params = ["T", "P", "rho", "Vp", "Vs", "melt_fraction", "assemblage",
                    "assemblage_variance"]

    # Convert numeric point results into numpy arrays
    for key, value in results.items():
        if key in point_params:
            if key == "P":
                # Convert from kbar to GPa
                results[key] = np.array(value) / 10

            elif key == "T":
                # Convert from C to K
                results[key] = np.array(value) + 273

            elif key == "rho":
                # Convert from kg/m3 to g/cm3
                results[key] = np.array(value) / 1000

            elif key == "melt_fraction":
                # Convert from kg/m3 to g/cm3
                results[key] = np.array(value) * 100

            else:
                results[key] = np.array(value)

    # Print results
    if verbose >= 2:
        units = {"T": "K", "P": "GPa", "rho": "g/cm3", "Vp": "km/s", "Vs": "km/s",
                 "melt_fraction": "%", "assemblage": "", "assemblage_variance": ""}

        print("+++++++++++++++++++++++++++++++++++++++++++++")
        for key, value in results.items():
            if isinstance(value, list):
                print(f"    ({len(value)},) list:      : {key}")

            elif isinstance(value, np.ndarray):
                min, max = np.nanmin(value), np.nanmax(value)

                print(
                    f"    {value.shape} np array: {key} ({min:.1f}, {max:.1f}) {units[key]}"
                )

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Save as pandas df
    df = pd.DataFrame.from_dict(results)

    if verbose >= 1:
        print(f"Writing MAGEMin results: {model_out_dir} ...")

    # Write to csv file
    df.to_csv(f"{model_out_dir}/results.csv", index=False)

    # Clean up output directory
    os.remove(f"{model_out_dir}/in.dat")
    shutil.rmtree(f"{model_out_dir}/output")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .3.1            Process Perple_X              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# read perplex output !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_perplex_output(filepath):
    """
    """
    # Initialize results
    results = {"T": [], "P": [], "rho": [], "Vp": [], "Vs": [], "entropy": [],
               "assemblage_index": [], "melt_fraction": [], "assemblage": [],
               "assemblage_variance": []}

    # Open file
    with open(filepath, "r") as file:
        # Skip lines until column headers are found
        for line in file:
            if line.strip().startswith("T(K)"):
                break

        # Read the data
        for line in file:
            # Split line on whitespace
            values = line.split()

            # Read the table of P, T, rho etc.
            if len(values) >= 8:
                try:
                    for i in range(8):
                        # Make values floats or assign nan
                        value = (float(values[i])
                                 if not np.isnan(float(values[i]))
                                 else np.nan)

                        # Convert from bar to GPa
                        if i == 1: # P column
                            value /= 1e4

                        # Convert assemblage index to an integer
                        if i == 6: # assemblage index column
                            value = int(value) if not np.isnan(value) else np.nan

                        # Convert from % to fraction
                        if i == 7: # melt fraction column
                            value /= 100

                        # Append results
                        results[list(results.keys())[i]].append(value)

                except ValueError:
                    continue
    return results

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# process perplex assemblage !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_perplex_assemblage(filepath):
    """
    """
    # Initialize dictionary to store assemblage info
    assemblage_dict = {}

    # Open assemblage file
    with open(filepath, "r") as file:
        for i, line in enumerate(file, start=1):
            assemblages = line.split("-")[1].strip().split()

            # Make string formatting consistent
            cleaned_assemblages = [
                assemblage.split("(")[0].lower() for assemblage in assemblages
            ]

            # Add assemblage to dict
            assemblage_dict[i] = cleaned_assemblages

    return assemblage_dict

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# process perplex results !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_perplex_results(sample_id, dataset, res, verbose):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Model output dir
    out_dir = "runs"

    # Get directory with perplex model outputs
    model_out_dir = f"{out_dir}/perplex_{sample_id}_{dataset}_{res}"

    # Check for Perple_X output files
    if not os.path.exists(f"{out_dir}"):
        sys.exit("No Perple_X files to process!")

    # Get filepaths for targets and assemblage files
    filepath_targets = f"{model_out_dir}/target-array.tab"
    filepath_assemblage = f"{model_out_dir}/assemblages.txt"

    # Check for targets
    if not os.path.exists(filepath_targets):
        sys.exit("No Perple_X target file to process ...")

    # Check for assemblages
    if not os.path.exists(filepath_assemblage):
        sys.exit("No Perple_X assemblage file to process ...")

    if verbose >= 1:
        print(f"Reading Perple_X output: {model_out_dir} ...")

    # Read results
    results = read_perplex_output(filepath_targets)

    # Remove entropy
    results.pop("entropy")

    # Get assemblages from file
    assemblages = process_perplex_assemblage(filepath_assemblage)

    # Parse assemblages by index
    for index in results.get("assemblage_index"):
        if np.isnan(index):
            results["assemblage"].append("")

        else:
            phases = assemblages[index]
            results["assemblage"].append(phases)

    # Count unique phases (assemblage variance)
    for assemblage in results.get("assemblage"):
        if assemblage is None:
            results["assemblage_variance"].append(np.nan)

        else:
            unique_phases = set(assemblage)
            count = len(unique_phases)

            results["assemblage_variance"].append(count)

    # Remove assemblage index
    results.pop("assemblage_index")

    # Encode assemblage
    filepath_csv = f"{model_out_dir}/assemblages.csv"
    encoded_assemblages = encode_assemblages(results["assemblage"], filepath_csv)

    # Replace assemblage with encoded assemblages
    results["assemblage"] = encoded_assemblages

    # Point results that can be converted to numpy arrays
    point_params = ["T", "P", "rho", "Vp", "Vs", "melt_fraction", "assemblage",
                    "assemblage_variance"]

    # Convert numeric point results into numpy arrays
    for key, value in results.items():
        if key in point_params:
            if key == "rho":
                # Convert from kg/m3 to g/cm3
                results[key] = np.array(value) / 1000

            elif key == "melt_fraction":
                # Convert from kg/m3 to g/cm3
                results[key] = np.array(value) * 100

            else:
                results[key] = np.array(value)

    # Print results
    if verbose >= 2:
        units = {"T": "K", "P": "GPa", "rho": "g/cm3", "Vp": "km/s", "Vs": "km/s",
                 "melt_fraction": "%", "assemblage": "", "assemblage_variance": ""}

        print("+++++++++++++++++++++++++++++++++++++++++++++")
        for key, value in results.items():
            if isinstance(value, list):
                print(f"    ({len(value)},) list       : {key}")

            elif isinstance(value, np.ndarray):
                min, max = np.nanmin(value), np.nanmax(value)

                print(
                    f"    {value.shape} np array: {key} ({min:.1f}, {max:.1f}) {units[key]}"
                )

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Save as pandas df
    df = pd.DataFrame.from_dict(results)

    if verbose >= 1:
        print(f"Writing Perple_X results: {model_out_dir} ...")

    # Write to csv file
    df.to_csv(f"{model_out_dir}/results.csv", index=False)

    # Clean up output directory
    files_to_keep = ["assemblages.csv", "results.csv"]

    try:
        # List all files in the directory
        all_files = os.listdir(model_out_dir)

        # Iterate through the files and delete those not in the exclusion list
        for filename in all_files:
            file_path = os.path.join(model_out_dir, filename)

            if os.path.isfile(file_path) and filename not in files_to_keep:
                os.remove(file_path)

    except Exception as e:
        print(f"Error: {e}")

#######################################################
## .4.                ML Methods                 !!! ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .4.0            Helper Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# scale arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def scale_arrays(features_array, targets_array):
    """
    """
    # Reshape the features and targets arrays
    X = features_array
    y = targets_array

    # Create nan mask
    mask = np.any([np.isnan(y[:,i]) for i in range(y.shape[1])], axis=0)

    # Remove nans
    X, y = X[~mask,:], y[~mask,:]

    # Initialize scalers
    scaler_X, scaler_y = StandardScaler(), StandardScaler()

    # Scale features array
    X_scaled = scaler_X.fit_transform(X)

    # Scale the target array
    y_scaled = scaler_y.fit_transform(y)

    return X, y, scaler_X, scaler_y, X_scaled, y_scaled

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# configure rocml model !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def configure_rocml_model(X_scaled, y_scaled, model, parallel, nprocs, tune, epochs, batchp,
                          seed, verbose):
    """
    """
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
        if nprocs is None or nprocs > os.cpu_count():
            nprocs = os.cpu_count() - 2

        else:
            nprocs = nprocs

    print(f"Configuring RocML model: {model_label_full} ...")

    if not tune:
        # Define ML models without tuning
        if model_label == "KN":
            model = KNeighborsRegressor(n_neighbors=4, weights="distance")

        elif model_label == "RF":
            model = RandomForestRegressor(random_state=seed, n_estimators=400,
                                          max_features=2, min_samples_leaf=1,
                                          min_samples_split=2)

        elif model_label == "DT":
            model = DecisionTreeRegressor(random_state=seed, splitter="best",
                                          max_features=2, min_samples_leaf=1,
                                          min_samples_split=2)

        elif model_label == "NN1":
            model = MLPRegressor(random_state=seed, max_iter=epochs,
                                 learning_rate_init=0.001,
                                 hidden_layer_sizes=(int(y_scaled.shape[0] * 0.1)))

        elif model_label == "NN2":
            model = MLPRegressor(random_state=seed, max_iter=epochs,
                                 learning_rate_init=0.0001,
                                 hidden_layer_sizes=(int(y_scaled.shape[0] * 0.5),
                                                     int(y_scaled.shape[0] * 0.2)))

        elif model_label == "NN3":
            model = MLPRegressor(random_state=seed, max_iter=epochs,
                                 learning_rate_init=0.0001,
                                 hidden_layer_sizes=(int(y_scaled.shape[0] * 0.5),
                                                     int(y_scaled.shape[0] * 0.2),
                                                     int(y_scaled.shape[0] * 0.1)))

    else:
        # Set verbose
        if verbose >= 2:
            verbose = True

        # Define ML model and grid search param space for hyperparameter tuning
        print("Tuning RocML model ...")

        if model_label == "KN":
            model = KNeighborsRegressor()

            param_grid = dict(n_neighbors=[2, 4, 8], weights=["uniform", "distance"])

        elif model_label == "RF":
            model = RandomForestRegressor(random_state=seed, verbose=verbose)

            param_grid = dict(n_estimators=[400, 800, 1200],
                              max_features=[1, 2, 3],
                              min_samples_leaf=[1, 2, 3],
                              min_samples_split=[2, 4, 6])

        elif model_label == "DT":
            model = DecisionTreeRegressor(random_state=seed)

            param_grid = dict(splitter=["best", "random"],
                              max_features=[1, 2, 3],
                              min_samples_leaf=[1, 2, 3],
                              min_samples_split=[2, 4, 6])

        elif model_label == "NN1":
            model = MLPRegressor(random_state=seed, max_iter=epochs, verbose=verbose,
                                 batch_size=max(int(len(y_scaled) * batchp), 8))

            param_grid = dict(hidden_layer_sizes=[(int(y_scaled.shape[0] * 0.1)),
                                                  (int(y_scaled.shape[0] * 0.2)),
                                                  (int(y_scaled.shape[0] * 0.5))],
                              learning_rate_init=[0.0001, 0.0005, 0.001])

        elif model_label == "NN2":
            model = MLPRegressor(random_state=seed, max_iter=epochs, verbose=verbose,
                                 batch_size=max(int(len(y_scaled) * batchp), 8))

            param_grid = dict(hidden_layer_sizes=[(int(y_scaled.shape[0] * 0.1),
                                                   int(y_scaled.shape[0] * 0.2)),
                                                  (int(y_scaled.shape[0] * 0.2),
                                                   int(y_scaled.shape[0] * 0.2)),
                                                  (int(y_scaled.shape[0] * 0.5),
                                                   int(y_scaled.shape[0] * 0.2))],
                              learning_rate_init=[0.0001, 0.0005, 0.001])

        elif model_label == "NN3":
            model = MLPRegressor(random_state=seed, max_iter=epochs, verbose=verbose,
                                 batch_size=max(int(len(y_scaled) * batchp), 8))

            param_grid = dict(hidden_layer_sizes=[(int(y_scaled.shape[0] * 0.1),
                                                   int(y_scaled.shape[0] * 0.2),
                                                   int(y_scaled.shape[0] * 0.1)),
                                                  (int(y_scaled.shape[0] * 0.2),
                                                   int(y_scaled.shape[0] * 0.2),
                                                   int(y_scaled.shape[0] * 0.1)),
                                                  (int(y_scaled.shape[0] * 0.5),
                                                   int(y_scaled.shape[0] * 0.2),
                                                   int(y_scaled.shape[0] * 0.1))],
                              learning_rate_init=[0.0001, 0.0005, 0.001])

        # K-fold cross-validation
        kf = KFold(n_splits=3, shuffle=True, random_state=seed)

        # Perform grid search hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf,
                                   scoring="neg_root_mean_squared_error",
                                   n_jobs=nprocs, verbose=1)

        grid_search.fit(X_scaled, y_scaled)

        print("Tuning done!")

        # Define ML model with tuned hyperparameters
        if model_label == "KN":
            model = KNeighborsRegressor(n_neighbors=grid_search.best_params_["n_neighbors"],
                                        weights=grid_search.best_params_["weights"])

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
                learning_rate_init=grid_search.best_params_["learning_rate_init"],
                hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"]
            )

    # Get hyperparameters
    model_hyperparams = model.get_params()

    print("Configuring done!")

    return model_label, model_label_full, model, model_hyperparams

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# iterate fold !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def iterate_fold(args):
    """
    """
    # Unpack arguments
    (train_index, test_index), X_scaled, y_scaled, X_scaled_val, y_scaled_val, model, \
        model_label, epochs, batchp, scaler_X, scaler_y, scaler_X_val, scaler_y_val, \
        fig_dir, verbose = args

    # Split the data into training and testing sets
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    X_val, y_val = X_scaled_val, y_scaled_val

    if "NN" in model_label:
        # Initialize lists to store loss values
        epoch_, train_loss_, valid_loss_ = [], [], []

        # Set batch size as a proportion of the training dataset size
        batch_size = int(len(y_train) * batchp)

        # Ensure a minimum batch size
        batch_size = max(batch_size, 8)

        # Start training timer
        training_start_time = time.time()

        # Partial training
        with tqdm(total=epochs, desc="Training NN", position=0) as pbar:
            for epoch in range(epochs):
                # Shuffle the training data for each epoch
                indices = np.arange(len(y_train))
                np.random.shuffle(indices)

                for start_idx in range(0, len(indices), batch_size):
                    end_idx = start_idx + batch_size

                    # Ensure that the batch size doesn't exceed the dataset size
                    end_idx = min(end_idx, len(indices))

                    # Subset training data
                    batch_indices = indices[start_idx:end_idx]
                    X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                    # Train NN model on batch
                    model.partial_fit(X_batch, y_batch)

                # Calculate and store training loss
                train_loss = model.loss_
                train_loss_.append(train_loss)

                # Calculate and store validation loss
                valid_loss = mean_squared_error(y_val, model.predict(X_val) / 2)
                valid_loss_.append(valid_loss)

                # Store epoch
                epoch_.append(epoch + 1)

                # Update progress bar
                pbar.update(1)

        # End training timer
        training_end_time = time.time()

        # Create loss curve dict
        loss_curve = {"epoch": epoch_, "train_loss": train_loss_, "valid_loss": valid_loss_}

    else:
        # Start training timer
        training_start_time = time.time()

        # Train ML model
        model.fit(X_train, y_train)

        # End training timer
        training_end_time = time.time()

        # Empty loss curve
        loss_curve = None

    # Calculate training time
    training_time = (training_end_time - training_start_time) * 1000

    # Make predictions on the test dataset
    y_pred_scaled = model.predict(X_test)
    y_pred_scaled_val = model.predict(X_val)

    # Test inference time on single random PT datapoint from the test dataset
    rand_PT_point = X_test[np.random.choice(X_test.shape[0], 1, replace=False)]

    inference_start_time = time.time()
    single_PT_pred = model.predict(rand_PT_point)
    inference_end_time = time.time()

    inference_time = (inference_end_time - inference_start_time) * 1000

    # Inverse transform predictions
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    y_pred_original_val = scaler_y_val.inverse_transform(y_pred_scaled_val)

    # Inverse transform test dataset
    y_test_original = scaler_y.inverse_transform(y_test)
    y_val_original = scaler_y_val.inverse_transform(y_val)

    # Calculate performance metrics to evaluate the model
    rmse_test = np.sqrt(mean_squared_error(y_test_original, y_pred_original,
                                           multioutput="raw_values"))

    rmse_val = np.sqrt(mean_squared_error(y_val_original, y_pred_original_val,
                                          multioutput="raw_values"))

    r2_test = r2_score(y_test_original, y_pred_original, multioutput="raw_values")
    r2_val = r2_score(y_val_original, y_pred_original_val, multioutput="raw_values")

    return loss_curve, rmse_test, r2_test, rmse_val, r2_val, training_time, inference_time

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# process_kfold_results !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_kfold_results(results, sample_id, model_label, model_label_full, program,
                          W, n_features, targets, units, kfolds, fig_dir, verbose):
    """
    """
    # Initialize empty lists for storing performance metrics
    loss_curves = []
    rmse_test_scores = []
    r2_test_scores = []
    rmse_val_scores = []
    r2_val_scores = []
    training_times = []
    inference_times = []

    # Unpack results
    for (loss_curve, rmse_test, r2_test, rmse_val, r2_val, training_time, inference_time
         ) in results:
        loss_curves.append(loss_curve)
        rmse_test_scores.append(rmse_test)
        r2_test_scores.append(r2_test)
        rmse_val_scores.append(rmse_val)
        r2_val_scores.append(r2_val)
        training_times.append(training_time)
        inference_times.append(inference_time)

    if "NN" in model_label:
        # Initialize empty dict for combined loss curves
        merged_curves = {}

        # Merge loss curves
        for curve in loss_curves:
            for key, value in curve.items():
                if key in merged_curves:
                    if isinstance(merged_curves[key], list):
                        merged_curves[key].extend(value)
                    else:
                        merged_curves[key] = [merged_curves[key], value]
                        merged_curves[key].extend(value)
                else:
                    merged_curves[key] = value

        # Make dict into pandas df
        df = pd.DataFrame.from_dict(merged_curves, orient="index").transpose()
        df.sort_values(by="epoch", inplace=True)

        # Set plot style and settings
        plt.rcParams["legend.facecolor"] = "0.9"
        plt.rcParams["legend.loc"] = "upper left"
        plt.rcParams["legend.fontsize"] = "small"
        plt.rcParams["legend.frameon"] = "False"
        plt.rcParams["axes.facecolor"] = "0.9"
        plt.rcParams["font.size"] = 12
        plt.rcParams["figure.autolayout"] = "True"
        plt.rcParams["figure.dpi"] = 330
        plt.rcParams["savefig.bbox"] = "tight"

        # Plot loss curve
        fig = plt.figure(figsize=(6.3, 3.54))

        # Colormap
        colormap = plt.cm.get_cmap("tab10")

        plt.plot(df["epoch"], df["train_loss"], label="train loss", color=colormap(0))
        plt.plot(df["epoch"], df["valid_loss"], label="valid loss", color=colormap(1))
        plt.xlabel("Epoch")
        plt.ylabel(f"Loss")

        if program == "magemin":
            program_label = "MAGEMin"

        if program == "perplex":
            program_label = "Perple_X"

        plt.title(f"{model_label_full} Loss Curve [{program_label}]")
        plt.legend()

        # Save the plot to a file if a filename is provided
        plt.savefig(f"{fig_dir}/{program}-{sample_id}-{model_label}-loss-curve.png")

        # Close plot
        plt.close()

    # Stack arrays
    rmse_test_scores = np.stack(rmse_test_scores)
    r2_test_scores = np.stack(r2_test_scores)
    rmse_val_scores = np.stack(rmse_val_scores)
    r2_val_scores = np.stack(r2_val_scores)

    # Calculate performance values with uncertainties
    rmse_test_mean = np.mean(rmse_test_scores, axis=0)
    rmse_test_std = np.std(rmse_test_scores, axis=0)
    r2_test_mean = np.mean(r2_test_scores, axis=0)
    r2_test_std = np.std(r2_test_scores, axis=0)
    rmse_val_mean = np.mean(rmse_val_scores, axis=0)
    rmse_val_std = np.std(rmse_val_scores, axis=0)
    r2_val_mean = np.mean(r2_val_scores, axis=0)
    r2_val_std = np.std(r2_val_scores, axis=0)
    training_time_mean = np.mean(training_times)
    training_time_std = np.std(training_times)
    inference_time_mean = np.mean(inference_times)
    inference_time_std = np.std(inference_times)

    # Config and performance info
    cv_info = {
        "sample": sample_id,
        "model": model_label,
        "program": program,
        "size": W,
        "n_targets": len(targets),
        "k_folds": kfolds,
        "training_time_mean": round(training_time_mean, 3),
        "training_time_std": round(training_time_std, 3),
        "inference_time_mean": round(inference_time_mean, 3),
        "inference_time_std": round(inference_time_std, 3)
    }

    # Add performance metrics for each parameter to the dictionary
    for i, target in enumerate(targets):
        cv_info[f"rmse_test_mean_{target}"] = round(rmse_test_mean[i], 3)
        cv_info[f"rmse_test_std_{target}"] = round(rmse_test_std[i], 3)
        cv_info[f"r2_test_mean_{target}"] = round(r2_test_mean[i], 3),
        cv_info[f"r2_test_std_{target}"] = round(r2_test_std[i], 3),
        cv_info[f"rmse_val_mean_{target}"] = round(rmse_val_mean[i], 3),
        cv_info[f"rmse_val_std_{target}"] = round(rmse_val_std[i], 3),
        cv_info[f"r2_val_mean_{target}"] = round(r2_val_mean[i], 3),
        cv_info[f"r2_val_std_{target}"] = round(r2_val_std[i], 3)

    if verbose >= 1:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # Print performance
        print(f"{model_label_full} performance:")
        print(f"    training time: {training_time_mean:.3f} ± {training_time_std:.3f}")
        print(f"    inference time: {inference_time_mean:.3f} ± {inference_time_std:.3f}")
        print(f"    rmse test:")
        for r, e, p, u in zip(rmse_test_mean, rmse_test_std, targets, units):
            print(f"        {p}: {r:.3f} ± {e:.3f} {u}")
        print(f"    r2 test:")
        for r, e, p in zip(r2_test_mean, r2_test_std, targets):
            print(f"        {p}: {r:.3f} ± {e:.3f}")
        print(f"    rmse valid:")
        for r, e, p, u in zip(rmse_val_mean, rmse_val_std, targets, units):
            print(f"        {p}: {r:.3f} ± {e:.3f} {u}")
        print(f"    r2 valid:")
        for r, e, p in zip(r2_val_mean, r2_val_std, targets):
            print(f"        {p}: {r:.3f} ± {e:.3f}")
        print("+++++++++++++++++++++++++++++++++++++++++++++")

    return cv_info

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# append to csv !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def append_to_csv(filepath, data_dict):
    """
    """
    # Check if the CSV file already exists
    if not pd.io.common.file_exists(filepath):
        df = pd.DataFrame(data_dict)

    else:
        df = pd.read_csv(filepath)

        # Append the new data dictionary to the DataFrame
        new_data = pd.DataFrame(data_dict)

        df = pd.concat([df, new_data], ignore_index=True)

    # Sort df
    df = df.sort_values(by=["sample", "model", "program", "size"])

    # Save the updated DataFrame back to the CSV file
    df.to_csv(filepath, index=False)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .4.1          PCA and Synthetic Sampling      !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pca mixing arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def pca_mixing_arrays(res, oxides, n_pca_components, k_pca_clusters, seed, verbose,
                      palette="tab10", figwidth=6.3, figheight=6.3, fontsize=22,
                      filename="earthchem-samples", fig_dir="figs"):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

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
    data = read_earthchem_data(oxides, verbose)

    # Sort by composition
    if "MGO" in oxides:
        data.sort_values(by=["SIO2", "MGO"], ascending=[True, False], inplace=True,
                         ignore_index=True)

    else:
        data.sort_values(by="SIO2", ascending=True, inplace=True, ignore_index=True)

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

    if verbose >= 1:
        # Print summary
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print("PCA summary:")
        print(f"    number of samples: {pca.n_samples_}")
        print(f"    PCA components: {n_pca_components}")
        print(f"    K-means clusters: {k_pca_clusters}")
        print(f"    features ({len(oxides)}): {oxides}")
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
    loadings = pd.DataFrame((pca.components_.T * np.sqrt(pca.explained_variance_)).T,
                            columns=oxides)

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
        ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(oxides))))
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

            scatter = ax.scatter(data.loc[indices, f"PC{n+1}"],
                                 data.loc[indices, f"PC{n+2}"], edgecolors="none",
                                 color=colormap(i), marker=".", label=comp)

        for oxide in oxides:
            ax.arrow(0, 0, loadings.at[n, oxide] * 3, loadings.at[n+1, oxide] * 3,
                     width=0.02, head_width=0.14, color="black")
            ax.text((loadings.at[n, oxide] * 3) + (loadings.at[n, oxide] * 1),
                    (loadings.at[n+1, oxide] * 3) + (loadings.at[n+1, oxide] * 1),
                    oxide, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.1),
                    fontsize=fontsize * 0.579, color="black", ha = "center", va = "center")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=0,
                  markerscale=3, handletextpad=-0.5, fontsize=fontsize * 0.694)
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

            scatter = ax.scatter(data.loc[indices, f"PC{n+1}"],
                                 data.loc[indices, f"PC{n+2}"], edgecolors="none",
                                 color=colormap(c+4), marker=".", alpha=0.3)

            clusters = ax.scatter(centroids[c, n], centroids[c, n+1], edgecolor="black",
                                  color=colormap(c+4), label=f"cluster {c+1}",
                                  marker="s", s=100)

            # Calculate mixing lines between cluster centroids
            if k_pca_clusters > 1:
                for i in range(c+1, k_pca_clusters):
                    m = ((centroids[i, n+1] - centroids[c, n+1]) /
                         (centroids[i, n] - centroids[c, n]))
                    b = centroids[c, n+1] - m * centroids[c, n]

                    x_vals = np.linspace(centroids[c, n], centroids[i, n], res)
                    y_vals = m * x_vals + b

                    ax.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=1)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, columnspacing=0,
                  handletextpad=-0.5, fontsize=fontsize * 0.694)
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
    print("Calculating mixing lines between cluster centroids ...")

    for n in range(n_pca_components):
        for c in range(k_pca_clusters):
            # Calculate mixing lines between cluster centroids
            if k_pca_clusters > 1:
                for i in range(c+1, k_pca_clusters):
                    if verbose >= 2:
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

                sns.scatterplot(data=synthetic_datasets[f"data_synthetic{i+1}{j+1}"],
                                x="SIO2", y=y, linewidth=0, s=15, color=".2", legend=False,
                                ax=ax, zorder=3)
                ax.annotate(f"{i+1}", xy=(first_element["SIO2"], first_element[y]),
                            xytext=(0, 0), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      edgecolor="black", facecolor="white", alpha=0.8),
                            fontsize=fontsize * 0.579, zorder=4)
                ax.annotate(f"{j+1}", xy=(last_element["SIO2"], last_element[y]),
                            xytext=(0, 0), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      edgecolor="black", facecolor="white", alpha=0.8),
                            fontsize=fontsize * 0.579, zorder=5)

        sns.kdeplot(data=data, x="SIO2", y=y, hue="COMPOSITION",
                    hue_order=["ultramafic", "mafic", "intermediate", "felsic"], fill=False,
                    ax=ax, levels=5, zorder=2)
        sns.scatterplot(data=data, x="SIO2", y=y, hue="COMPOSITION",
                        hue_order=["ultramafic", "mafic", "intermediate", "felsic"],
                        linewidth=0, s=5, legend=False, ax=ax, zorder=1)

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

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4)
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
#+ .4.1      ML Training and Cross-Validation    !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cv rocml !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cv_rocml(features_array, targets_array, features_array_val, targets_array_val, targets,
             units, program, sample_id, model, tune, epochs, batchp, seed, kfolds, parallel,
             nprocs, vmin, vmax, palette, fig_dir, filename, verbose, figwidth=6.3,
             figheight=4.725, fontsize=22):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Get unit labels
    if units is None:
        units_labels = ["" for unit in units]

    else:
        units_labels = [f"({unit})" for unit in units]

    # Scale training dataset
    X, y, scaler_X, scaler_y, X_scaled, y_scaled = scale_arrays(features_array,
                                                                targets_array)

    # Scale validation dataset
    X_val, y_val, scaler_X_val, scaler_y_val, \
        X_scaled_val, y_scaled_val = scale_arrays(features_array_val, targets_array_val)

    # Define ML model
    model_label, model_label_full, \
        model, model_hyperparams = configure_rocml_model(X_scaled, y_scaled, model, parallel,
                                                         nprocs, tune, epochs, batchp, seed,
                                                         verbose)

    # Print rocml model config
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("Training RocML model:")
    print(f"    program: {program}")
    print(f"    sample: {sample_id}")
    print(f"    model: {model_label_full}")
    if "NN" in model_label:
        print(f"    epochs: {epochs}")
    print(f"    k folds: {kfolds}")
    print(f"    features:")
    print(f"        Pressure (GPa)")
    print(f"        Temperature (K)")
    print(f"    targets:")
    for target, unit in zip(targets, units_labels):
        print(f"        {target} {unit}")
    print(f"    features array shape: {features_array.shape}")
    print(f"    targets array shape: {targets_array.shape}")
    print(f"    hyperparameters:")
    for key, value in model_hyperparams.items():
        print(f"        {key}: {value}")
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Running kfold ({kfolds}) cross validation ...")

    # K-fold cross-validation
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

    # Create list of args for mp pooling
    fold_args = [
        (
            (train_index, test_index), X_scaled, y_scaled, X_scaled_val, y_scaled_val, model,
            model_label, epochs, batchp, scaler_X, scaler_y, scaler_X_val, scaler_y_val,
            fig_dir, verbose
        )
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X))
    ]

    # Create a multiprocessing pool
    with mp.Pool(processes=nprocs) as pool:
        results = pool.map(iterate_fold, fold_args)

        # Wait for all processes
        pool.close()
        pool.join()

    print("Kfold cross validation done!")

    # Get number of features and targets
    n_features = features_array.shape[-1]
    n_targets = targets_array.shape[-1]

    # Get length of features and targets
    W = features_array.shape[0]

    # Get cv info
    cv_info = process_kfold_results(results, sample_id, model_label, model_label_full,
                                    program, W, n_features, targets, units, kfolds,
                                    fig_dir, verbose)

    # Visualize cv results
    visualize_cv_results(program, model, model_label, model_label_full, targets,
                         features_array_val, targets_array_val, scaler_X_val, scaler_y_val,
                         X_scaled, y_scaled, cv_info, seed, palette, vmin, vmax, fig_dir,
                         filename, figwidth, figheight, fontsize)

    return model, cv_info

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train rocml !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_rocml(program, sample_id, res, targets, mask_geotherm, model, tune, epochs, batchp,
                kfolds, parallel, nprocs, seed, palette, fig_dir, verbose):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Model output dir
    out_dir = "runs"

    # Check for model output files
    if not os.path.exists(f"{out_dir}"):
        sys.exit(f"No {program} files to process!")

    # Check for incorrect program argument
    if program not in ["magemin", "perplex"]:
        sys.exit("program argument must be either magemin or perplex!")

    # Ignore the ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Modify model string
    model_label = model.replace(" ", "-")

    # Get model results
    if program == "magemin":
        results = read_gfem_results(program, sample_id, "train", res, verbose)
        results_val = read_gfem_results(program, sample_id, "valid", res, verbose)

    elif program == "perplex":
        results = read_gfem_results(program, sample_id, "train", res, verbose)
        results_val = read_gfem_results(program, sample_id, "valid", res, verbose)

    # Get training dataset
    features_array = create_feature_array(results, False)
    targets_array = create_target_array(results, targets, mask_geotherm)

    # Get validation dataset
    features_array_val = create_feature_array(results, False)
    targets_array_val = create_target_array(results, targets, mask_geotherm)

    # Create empty lists
    units, vmin, vmax = [], [], []

    # Get units and colorbar limits
    for i, target in enumerate(targets):
        # Units
        if target == "rho":
            units.append("g/cm$^3$")

        if target in ["Vp", "Vs"]:
            units.append("km/s")

        if target == "melt_fraction":
            units.append("%")

        # Get min max of target array
        vmin.append(np.nanmin(targets_array[:, i]))
        vmax.append(np.nanmax(targets_array[:, i]))

    # Create filename for output
    fname = f"{program}-{sample_id}-{model_label}"

    # Train models, predict, analyze
    model, info = cv_rocml(features_array, targets_array, features_array_val,
                           targets_array_val, targets, units, program, sample_id, model,
                           tune, epochs, batchp, seed, kfolds, parallel, nprocs, vmin, vmax,
                           palette, fig_dir, fname, verbose)

    # Write ML model config and performance info to csv
    append_to_csv(f"{data_dir}/rocml-performance.csv", info)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#######################################################
## .5.              Visualizations               !!! ##
#######################################################

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .5.1            Helper Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get geotherm !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_geotherm(results, target, threshold, thermal_gradient=0.5, mantle_potential_T=1573):
    """
    """
    # Get PT and target values and transform units
    df = pd.DataFrame({"P": results["P"], "T": results["T"],
                       target: results[target]}).sort_values(by="P")

    # Calculate geotherm
    df["geotherm_P"] = (df["T"] - mantle_potential_T) / (thermal_gradient * 35)

    # Subset df along geotherm
    df_geotherm = df[abs(df["P"] - df["geotherm_P"]) < threshold]

    # Extract the three vectors
    P_values = df_geotherm["P"].values
    T_values = df_geotherm["T"].values
    targets = df_geotherm[target].values

    return P_values, T_values, targets

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine plots horizontally !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def combine_plots_horizontally(image1_path, image2_path, output_path, caption1, caption2,
                               font_size=150, caption_margin=25, dpi=330):
    """
    """
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Determine the maximum height between the two images
    max_height = max(image1.height, image2.height)

    # Create a new image with twice the width and the maximum height
    combined_width = image1.width + image2.width
    combined_image = Image.new("RGB", (combined_width, max_height), (255, 255, 255))

    # Set the DPI metadata
    combined_image.info["dpi"] = (dpi, dpi)

    # Paste the first image on the left
    combined_image.paste(image1, (0, 0))

    # Paste the second image on the right
    combined_image.paste(image2, (image1.width, 0))

    # Add captions
    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("Arial", font_size)
    caption_margin = caption_margin

    # Add caption
    draw.text((caption_margin, caption_margin), caption1, font=font, fill="black")

    # Add caption "b"
    draw.text((image1.width + caption_margin, caption_margin), caption2, font=font,
              fill="black")

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# combine plots vertically !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def combine_plots_vertically(image1_path, image2_path, output_path, caption1, caption2,
                             font_size=150, caption_margin=25, dpi=330):
    """
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

    # Add caption
    draw.text((caption_margin, caption_margin), caption1, font=font, fill="black")

    # Add caption "b"
    draw.text((caption_margin, image1.height + caption_margin), caption2, font=font,
              fill="black")

    # Save the combined image with captions
    combined_image.save(output_path, dpi=(dpi, dpi))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose dataset plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_dataset_plots(magemin, perplex, sample_id, dataset, res, targets, fig_dir,
                          verbose):
    """
    """
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

    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    print("Composing plots ...")

    # Compose plots
    if magemin and perplex:
        for target in targets_rename:
            combine_plots_horizontally(
                f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png",
                caption1="a)",
                caption2="b)"
            )

            if target not in ["assemblage", "assemblage-variance"]:
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                    f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                    f"{fig_dir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                combine_plots_horizontally(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/diff-{sample_id}-{dataset}-{target}.png",
                    f"{fig_dir}/image3-{sample_id}-{dataset}-{target}.png",
                    caption1="",
                    caption2="c)"
                )

                os.remove(f"{fig_dir}/temp1.png")

            # PREM variables
            if target in ["rho", "Vp", "Vs"]:
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                    f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                    f"{fig_dir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                combine_plots_horizontally(
                    f"{fig_dir}/diff-{sample_id}-{dataset}-{target}.png",
                    f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                    f"{fig_dir}/temp2.png",
                    caption1="c)",
                    caption2="d)"
                )

                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp2.png",
                    f"{fig_dir}/image4-{sample_id}-{dataset}-{target}.png",
                    caption1="",
                    caption2=""
                )

                os.remove(f"{fig_dir}/temp1.png")
                os.remove(f"{fig_dir}/temp2.png")
                os.remove(f"{fig_dir}/diff-{sample_id}-{dataset}-{target}.png")
                os.remove(f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png")

            if target == "melt-fraction":
                os.remove(f"{fig_dir}/diff-{sample_id}-{dataset}-{target}.png")

            os.remove(f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png")
            os.remove(f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png")

    elif magemin and not perplex:
        # Get magemin results
        results_mgm = process_magemin_results(sample_id, dataset, res, verbose)

        for target in targets_rename:
            if target not in ["assemblage", "assemblage-variance"]:
                if target in ["rho", "Vp", "Vs"]:
                    if target == "rho":
                        visualize_prem(target, "g/cm$^3$", results_mgm, None,
                                       geotherm_threshold=geotherm_threshold,
                                       title="PREM Comparison", fig_dir=fig_dir,
                                       filename=f"prem-{sample_id}-{dataset}-{target}.png")

                    if target in ["Vp", "Vs"]:
                        visualize_prem(target, "km/s", results_mgm, None,
                                       geotherm_threshold=geotherm_threshold,
                                       title="PREM Comparison", fig_dir=fig_dir,
                                       filename=f"prem-{sample_id}-{dataset}-{target}.png")

                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image2-{sample_id}-{dataset}-{target}.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    os.remove(f"{fig_dir}/magemin-{sample_id}-{dataset}-{target}.png")
                    os.remove(f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png")

    elif perplex and not magemin:
        # Get perplex results
        results_ppx = process_perplex_results(sample_id, dataset, res, verbose)

        for target in targets_rename:
            if target not in ["assemblage", "assemblage-variance"]:
                if target in ["rho", "Vp", "Vs"]:
                    if target == "rho":
                        visualize_prem(target, "g/cm$^3$", None, results_ppx,
                                       geotherm_threshold=geotherm_threshold,
                                       title="PREM Comparison", fig_dir=fig_dir,
                                       filename=f"prem-{sample_id}-{dataset}-{target}.png")

                    if target in ["Vp", "Vs"]:
                        visualize_prem(target, "km/s", None, results_ppx,
                                       geotherm_threshold=geotherm_threshold,
                                       title="PREM Comparison", fig_dir=fig_dir,
                                       filename=f"prem-{sample_id}-{dataset}-{target}.png")

                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png",
                        f"{fig_dir}/image-{sample_id}-{dataset}-{target}.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    os.remove(f"{fig_dir}/perplex-{sample_id}-{dataset}-{target}.png")
                    os.remove(f"{fig_dir}/prem-{sample_id}-{dataset}-{target}.png")

    print("Composing done!")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# compose rocml plots !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def compose_rocml_plots(magemin, perplex, sample_id, models, targets, fig_dir):
    """
    """
    # Rename targets
    targets_rename = [target.replace("_", "-") for target in targets]

    print("Composing plots ...")

    # Compose plots
    for model in models:
        if "NN" in model:
            # First row
            combine_plots_vertically(
                f"{fig_dir}/magemin-{sample_id}-{model}-loss-curve.png",
                f"{fig_dir}/perplex-{sample_id}-{model}-loss-curve.png",
                f"{fig_dir}/loss-{sample_id}-{model}.png",
                caption1="a)",
                caption2="b)"
            )

        for target in targets_rename:
            if target in ["rho", "Vp", "Vs"]:
                # First row
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{model}-{target}-prem.png",
                    f"{fig_dir}/perplex-{sample_id}-{model}-{target}-prem.png",
                    f"{fig_dir}/prem-{sample_id}-{model}-{target}.png",
                    caption1="a)",
                    caption2="b)"
                )

            # First row
            combine_plots_horizontally(
                f"{fig_dir}/magemin-{sample_id}-{model}-{target}-targets-surf.png",
                f"{fig_dir}/perplex-{sample_id}-{model}-{target}-targets-surf.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            # Second row
            combine_plots_horizontally(
                f"{fig_dir}/magemin-{sample_id}-{model}-{target}-surf.png",
                f"{fig_dir}/perplex-{sample_id}-{model}-{target}-surf.png",
                f"{fig_dir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            # Third row
            combine_plots_horizontally(
                f"{fig_dir}/magemin-{sample_id}-{model}-{target}-diff-surf.png",
                f"{fig_dir}/perplex-{sample_id}-{model}-{target}-diff-surf.png",
                f"{fig_dir}/temp4.png",
                caption1="e)",
                caption2="f)"
            )

            # Stack rows
            combine_plots_vertically(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/temp2.png",
                f"{fig_dir}/temp3.png",
                caption1="",
                caption2=""
            )

            # Stack rows
            combine_plots_vertically(
                f"{fig_dir}/temp3.png",
                f"{fig_dir}/temp4.png",
                f"{fig_dir}/surf-{sample_id}-{model}-{target}.png",
                caption1="",
                caption2=""
            )

            # First row
            combine_plots_horizontally(
                f"{fig_dir}/magemin-{sample_id}-{model}-{target}-targets.png",
                f"{fig_dir}/perplex-{sample_id}-{model}-{target}-targets.png",
                f"{fig_dir}/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            # Second row
            combine_plots_horizontally(
                f"{fig_dir}/magemin-{sample_id}-{model}-{target}-predictions.png",
                f"{fig_dir}/perplex-{sample_id}-{model}-{target}-predictions.png",
                f"{fig_dir}/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            # Third row
            combine_plots_horizontally(
                f"{fig_dir}/magemin-{sample_id}-{model}-{target}-diff.png",
                f"{fig_dir}/perplex-{sample_id}-{model}-{target}-diff.png",
                f"{fig_dir}/temp4.png",
                caption1="e)",
                caption2="f)"
            )

            # Stack rows
            combine_plots_vertically(
                f"{fig_dir}/temp1.png",
                f"{fig_dir}/temp2.png",
                f"{fig_dir}/temp3.png",
                caption1="",
                caption2=""
            )

            # Stack rows
            combine_plots_vertically(
                f"{fig_dir}/temp3.png",
                f"{fig_dir}/temp4.png",
                f"{fig_dir}/image6-{sample_id}-{model}-{target}.png",
                caption1="",
                caption2=""
            )

            os.remove(f"{fig_dir}/temp1.png")
            os.remove(f"{fig_dir}/temp2.png")
            os.remove(f"{fig_dir}/temp3.png")
            os.remove(f"{fig_dir}/temp4.png")

            if len(models) == 6:
                # First row
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{models[0]}-{target}-surf.png",
                    f"{fig_dir}/magemin-{sample_id}-{models[1]}-{target}-surf.png",
                    f"{fig_dir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                # Second row
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{models[2]}-{target}-surf.png",
                    f"{fig_dir}/magemin-{sample_id}-{models[3]}-{target}-surf.png",
                    f"{fig_dir}/temp2.png",
                    caption1="c)",
                    caption2="d)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp2.png",
                    f"{fig_dir}/temp3.png",
                    caption1="",
                    caption2=""
                )

                # Third row
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{models[4]}-{target}-surf.png",
                    f"{fig_dir}/magemin-{sample_id}-{models[5]}-{target}-surf.png",
                    f"{fig_dir}/temp4.png",
                    caption1="e)",
                    caption2="f)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp3.png",
                    f"{fig_dir}/temp4.png",
                    f"{fig_dir}/magemin-{sample_id}-{target}-comp-surf.png",
                    caption1="",
                    caption2=""
                )

                # First row
                combine_plots_horizontally(
                    f"{fig_dir}/perplex-{sample_id}-{models[0]}-{target}-surf.png",
                    f"{fig_dir}/perplex-{sample_id}-{models[1]}-{target}-surf.png",
                    f"{fig_dir}/temp1.png",
                    caption1="g)",
                    caption2="h)"
                )

                # Second row
                combine_plots_horizontally(
                    f"{fig_dir}/perplex-{sample_id}-{models[2]}-{target}-surf.png",
                    f"{fig_dir}/perplex-{sample_id}-{models[3]}-{target}-surf.png",
                    f"{fig_dir}/temp2.png",
                    caption1="i)",
                    caption2="j)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp2.png",
                    f"{fig_dir}/temp3.png",
                    caption1="",
                    caption2=""
                )

                # Third row
                combine_plots_horizontally(
                    f"{fig_dir}/perplex-{sample_id}-{models[4]}-{target}-surf.png",
                    f"{fig_dir}/perplex-{sample_id}-{models[5]}-{target}-surf.png",
                    f"{fig_dir}/temp4.png",
                    caption1="k)",
                    caption2="l)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp3.png",
                    f"{fig_dir}/temp4.png",
                    f"{fig_dir}/perplex-{sample_id}-{target}-comp-surf.png",
                    caption1="",
                    caption2=""
                )

                # Stack rows
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{target}-comp-surf.png",
                    f"{fig_dir}/perplex-{sample_id}-{target}-comp-surf.png",
                    f"{fig_dir}/all-surf-{sample_id}-{target}.png",
                    caption1="",
                    caption2=""
                )

                # First row
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{models[0]}-{target}-predictions.png",
                    f"{fig_dir}/magemin-{sample_id}-{models[1]}-{target}-predictions.png",
                    f"{fig_dir}/temp1.png",
                    caption1="a)",
                    caption2="b)"
                )

                # Second row
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{models[2]}-{target}-predictions.png",
                    f"{fig_dir}/magemin-{sample_id}-{models[3]}-{target}-predictions.png",
                    f"{fig_dir}/temp2.png",
                    caption1="c)",
                    caption2="d)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp2.png",
                    f"{fig_dir}/temp3.png",
                    caption1="",
                    caption2=""
                )

                # Third row
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{models[4]}-{target}-predictions.png",
                    f"{fig_dir}/magemin-{sample_id}-{models[5]}-{target}-predictions.png",
                    f"{fig_dir}/temp4.png",
                    caption1="e)",
                    caption2="f)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp3.png",
                    f"{fig_dir}/temp4.png",
                    f"{fig_dir}/magemin-{sample_id}-{target}-comp-image.png",
                    caption1="",
                    caption2=""
                )

                # First row
                combine_plots_horizontally(
                    f"{fig_dir}/perplex-{sample_id}-{models[0]}-{target}-predictions.png",
                    f"{fig_dir}/perplex-{sample_id}-{models[1]}-{target}-predictions.png",
                    f"{fig_dir}/temp1.png",
                    caption1="g)",
                    caption2="h)"
                )

                # Second row
                combine_plots_horizontally(
                    f"{fig_dir}/perplex-{sample_id}-{models[2]}-{target}-predictions.png",
                    f"{fig_dir}/perplex-{sample_id}-{models[3]}-{target}-predictions.png",
                    f"{fig_dir}/temp2.png",
                    caption1="i)",
                    caption2="j)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp1.png",
                    f"{fig_dir}/temp2.png",
                    f"{fig_dir}/temp3.png",
                    caption1="",
                    caption2=""
                )

                # Third row
                combine_plots_horizontally(
                    f"{fig_dir}/perplex-{sample_id}-{models[4]}-{target}-predictions.png",
                    f"{fig_dir}/perplex-{sample_id}-{models[5]}-{target}-predictions.png",
                    f"{fig_dir}/temp4.png",
                    caption1="k)",
                    caption2="l)"
                )

                # Stack rows
                combine_plots_vertically(
                    f"{fig_dir}/temp3.png",
                    f"{fig_dir}/temp4.png",
                    f"{fig_dir}/perplex-{sample_id}-{target}-comp-image.png",
                    caption1="",
                    caption2=""
                )

                # Stack rows
                combine_plots_horizontally(
                    f"{fig_dir}/magemin-{sample_id}-{target}-comp-image.png",
                    f"{fig_dir}/perplex-{sample_id}-{target}-comp-image.png",
                    f"{fig_dir}/all-image-{sample_id}-{target}.png",
                    caption1="",
                    caption2=""
                )

                if target in ["rho", "Vp", "Vs"]:
                    # First row
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{models[0]}-{target}-prem.png",
                        f"{fig_dir}/magemin-{sample_id}-{models[1]}-{target}-prem.png",
                        f"{fig_dir}/temp1.png",
                        caption1="a)",
                        caption2="b)"
                    )

                    # Second row
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{models[2]}-{target}-prem.png",
                        f"{fig_dir}/magemin-{sample_id}-{models[3]}-{target}-prem.png",
                        f"{fig_dir}/temp2.png",
                        caption1="c)",
                        caption2="d)"
                    )

                    # Stack rows
                    combine_plots_vertically(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/temp2.png",
                        f"{fig_dir}/temp3.png",
                        caption1="",
                        caption2=""
                    )

                    # Third row
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{models[4]}-{target}-prem.png",
                        f"{fig_dir}/magemin-{sample_id}-{models[5]}-{target}-prem.png",
                        f"{fig_dir}/temp4.png",
                        caption1="e)",
                        caption2="f)"
                    )

                    # Stack rows
                    combine_plots_vertically(
                        f"{fig_dir}/temp3.png",
                        f"{fig_dir}/temp4.png",
                        f"{fig_dir}/magemin-{sample_id}-{target}-comp-prem.png",
                        caption1="",
                        caption2=""
                    )

                    # First row
                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{models[0]}-{target}-prem.png",
                        f"{fig_dir}/perplex-{sample_id}-{models[1]}-{target}-prem.png",
                        f"{fig_dir}/temp1.png",
                        caption1="g)",
                        caption2="h)"
                    )

                    # Second row
                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{models[2]}-{target}-prem.png",
                        f"{fig_dir}/perplex-{sample_id}-{models[3]}-{target}-prem.png",
                        f"{fig_dir}/temp2.png",
                        caption1="i)",
                        caption2="j)"
                    )

                    # Stack rows
                    combine_plots_vertically(
                        f"{fig_dir}/temp1.png",
                        f"{fig_dir}/temp2.png",
                        f"{fig_dir}/temp3.png",
                        caption1="",
                        caption2=""
                    )

                    # Third row
                    combine_plots_horizontally(
                        f"{fig_dir}/perplex-{sample_id}-{models[4]}-{target}-prem.png",
                        f"{fig_dir}/perplex-{sample_id}-{models[5]}-{target}-prem.png",
                        f"{fig_dir}/temp4.png",
                        caption1="k)",
                        caption2="l)"
                    )

                    # Stack rows
                    combine_plots_vertically(
                        f"{fig_dir}/temp3.png",
                        f"{fig_dir}/temp4.png",
                        f"{fig_dir}/perplex-{sample_id}-{target}-comp-prem.png",
                        caption1="",
                        caption2=""
                    )

                    # Stack rows
                    combine_plots_horizontally(
                        f"{fig_dir}/magemin-{sample_id}-{target}-comp-prem.png",
                        f"{fig_dir}/perplex-{sample_id}-{target}-comp-prem.png",
                        f"{fig_dir}/all-prem-{sample_id}-{target}.png",
                        caption1="",
                        caption2=""
                    )

    # Clean up directory
    tmp_files = glob.glob(f"{fig_dir}/temp*.png")
    mgm_files = glob.glob(f"{fig_dir}/magemin*.png")
    ppx_files = glob.glob(f"{fig_dir}/perplex*.png")

    for file in tmp_files + mgm_files + ppx_files:
        os.remove(file)

    print("Composing done!")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ .5.2          Plotting Functions              !!! ++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize training dataset design !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_training_dataset_design(P_min, P_max, T_min, T_max, T_mantle1=273,
                                      T_mantle2=1773, grad_mantle1=1, grad_mantle2=0.5,
                                      fig_dir="figs", fontsize=12, figwidth=6.3,
                                      figheight=3.54):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # T range
    T = np.arange(0, T_max + 728)

    # Olivine --> Ringwoodite Clapeyron slopes
    references_410 = {"[410] Akaogi89": [0.001, 0.002], "[410] Katsura89": [0.0025],
                      "[410] Morishima94": [0.0034, 0.0038]}

    # Ringwoodite --> Bridgmanite + Ferropericlase Clapeyron slopes
    references_660 = {"[660] Ito82": [-0.002], "[660] Ito89 & Hirose02": [-0.0028],
                      "[660] Ito90": [-0.002, -0.006], "[660] Katsura03": [-0.0004, -0.002],
                      "[660] Akaogi07": [-0.0024, -0.0028]}

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
    colormap = plt.cm.get_cmap("tab10")
    colors = [colormap(i) for i in range(9)]

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

            plt.plot(T[(T >= 1200) & (T <= 2000)], line[(T >= 1200) & (T <= 2000)],
                     color=color, label=label)

            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Ringwoodite --> Bridgmanite + Ferropericlase
    for j, (ref, ref_lines) in enumerate(zip(references_660.keys(), lines_660)):
        color = colors[j + i + 1 % len(colors)]

        for j, line in enumerate(ref_lines):
            label = f"{ref}" if j == 0 else None

            plt.plot(T[(T >= 1200) & (T <= 2000)], line[(T >= 1200) & (T <= 2000)],
                     color=color, label=label)

            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Plot shaded rectangle for PT range of training dataset
    fill = plt.fill_between(T, P_min, P_max, where=(T >= T_min) & (T <= T_max),
                            color="gray", alpha=0.2)

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
    plt.fill(vertices[:, 0], vertices[:, 1], facecolor="blue", edgecolor="black", alpha=0.1)

    # Geotherm legend handles
    geotherm1_handle = mlines.Line2D([], [], linestyle="-", color="black",
                                     label="Geotherm 1")
    geotherm2_handle = mlines.Line2D([], [], linestyle="--", color="black",
                                     label="Geotherm 2")

    # Phase boundaries legend handles
    ref_line_handles = [
        mlines.Line2D([], [], color=color, label=label)
        for label, color in label_color_mapping.items() if label
    ]

    # Add geotherms to legend handles
    ref_line_handles.extend([geotherm1_handle, geotherm2_handle])

    db_data_handle = mpatches.Patch(color="gray", alpha=0.2, label="Training Data Range")

    labels_660.add("Training Data Range")
    label_color_mapping["Training Data Range"] = "gray"

    training_data_handle = mpatches.Patch(facecolor="blue", edgecolor="black", alpha=0.1,
                                          label="Mantle Conditions")

    labels_660.add("Mantle Conditions")
    label_color_mapping["Mantle Conditions"] = "gray"

    # Define the desired order of the legend items
    desired_order = ["Training Data Range", "Mantle Conditions", "[410] Akaogi89",
                     "[410] Katsura89", "[410] Morishima94", "[660] Ito82",
                     "[660] Ito89 & Hirose02", "[660] Ito90", "[660] Katsura03",
                     "[660] Akaogi07", "Geotherm 1", "Geotherm 2"]

    # Sort the legend handles based on the desired order
    legend_handles = sorted(ref_line_handles + [db_data_handle, training_data_handle],
                            key=lambda x: desired_order.index(x.get_label()))

    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (GPa)")
    plt.title("RocML Traning Dataset Design")
    plt.xlim(T_min - 100, T_max + 100)
    plt.ylim(P_min - 1, P_max + 1)

    # Move the legend outside the plot to the right
    plt.legend(title="", handles=legend_handles, loc="center left",
               bbox_to_anchor=(1.02, 0.5))

    # Adjust the figure size
    fig = plt.gcf()
    fig.set_size_inches(figwidth, figheight)

    # Save the plot to a file
    plt.savefig(f"{fig_dir}/training-dataset-design.png")

    # Close device
    plt.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize benchmark efficiency !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_benchmark_efficiency(fig_dir, filename, fontsize=12, figwidth=6.3,
                                   figheight=3.54):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read data
    data = pd.read_csv(f"{data_dir}/benchmark-efficiency.csv")

    # Arrange data by resolution and sample
    data.sort_values(by=["size", "sample", "program"], inplace=True)

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
    colormap = plt.cm.get_cmap("tab10")
    sample_colors = {"DMM": colormap(0), "NMORB": colormap(1),
                     "PUM": colormap(2), "RE46": colormap(3)}

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
        mgm_x = mgm_data["size"]
        mgm_y = mgm_data["time"]

        # Filter out rows with missing time values for ppx column
        ppx_data = group_data[group_data["program"] == "perplex"]
        ppx_x = ppx_data["size"]
        ppx_y = ppx_data["time"]

        # Plot mgm data points and connect them with lines
        line_mgm, = plt.plot(mgm_x, mgm_y, marker="o", color=color_val,
                             linestyle="-", label=f"[MAGEMin] {sample_val}")

        legend_handles.append(line_mgm)
        legend_labels.append(f"[MAGEMin] {sample_val}")

        # Plot ppx data points and connect them with lines
        line_ppx, = plt.plot(ppx_x, ppx_y, marker="s", color=color_val,
                             linestyle="--", label=f"[Perple_X] {sample_val}")

        legend_handles.append(line_ppx)
        legend_labels.append(f"[Perple_X] {sample_val}")

    # Set labels and title
    plt.xlabel("Number of Minimizations (PT Points)")
    plt.ylabel("Elapsed Time (s)")
    plt.title("Solution Efficiency")
    plt.xscale("log")
    plt.yscale("log")

    # Create the legend with the desired order
    plt.legend(legend_handles, legend_labels, title="",
               bbox_to_anchor=(1.02, 0.5), loc="center left")

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize prem !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_prem(target, target_unit, results_mgm=None, results_ppx=None,
                   results_ml=None, model=None, geotherm_threshold=0.1, metrics=None,
                   title=None, fig_dir="figs", filename=None, figwidth=6.3, figheight=4.725,
                   fontsize=22):
    """
    """
    # Data asset dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read the CSV file into a pandas DataFrame
    df_prem = pd.read_csv(f"{data_dir}/prem.csv")

    # Extract depth and target values
    target_prem = df_prem[target]
    depth_prem = df_prem["depth"]

    # Transform depth to pressure
    P_prem = depth_prem / 30

    # Initialize geotherms
    P_mgm, P_ppx, P_ml = None, None, None
    target_mgm, target_ppx, target_ml = None, None, None

    # Extract target values along a geotherm
    if results_mgm:
        P_mgm, _, target_mgm = get_geotherm(results_mgm, target, geotherm_threshold)

    if results_ppx:
        P_ppx, _, target_ppx = get_geotherm(results_ppx, target, geotherm_threshold)

    if results_ml:
        P_ml, _, target_ml = get_geotherm(results_ml, target, geotherm_threshold)

    # Get min and max P from geotherms
    P_min = min(np.nanmin(P) for P in [P_mgm, P_ppx, P_ml] if P is not None)
    P_max = max(np.nanmax(P) for P in [P_mgm, P_ppx, P_ml] if P is not None)

    # Create cropping mask for prem
    mask_prem = (P_prem >= P_min) & (P_prem <= P_max)

    # Crop pressure and target values
    P_prem, target_prem = P_prem[mask_prem], target_prem[mask_prem]

    # Crop results
    if results_mgm:
        mask_mgm = (P_mgm >= P_min) & (P_mgm <= P_max)
        P_mgm, target_mgm = P_mgm[mask_mgm], target_mgm[mask_mgm]

    if results_ppx:
        mask_ppx = (P_ppx >= P_min) & (P_ppx <= P_max)
        P_ppx, target_ppx = P_ppx[mask_ppx], target_ppx[mask_ppx]

    if results_ml:
        mask_ml = (P_ml >= P_min) & (P_ml <= P_max)
        P_ml, target_ml = P_ml[mask_ml], target_ml[mask_ml]

    # Get min max
    target_min = min(min(np.nanmin(lst) for lst in [target_mgm, target_ppx, target_ml]
                         if lst is not None), min(target_prem))
    target_max = max(max(np.nanmax(lst) for lst in [target_mgm, target_ppx, target_ml]
                         if lst is not None), max(target_prem))

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
    colormap = plt.cm.get_cmap("tab10")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(figwidth, figheight))

    # Plot PREM data on the primary y-axis
    ax1.plot(target_prem, P_prem, "-", linewidth=3, color="black", label="PREM")

    if results_mgm:
        ax1.plot(target_mgm, P_mgm, "-", linewidth=3, color=colormap(0), label="MAGEMin")

    if results_ppx:
        ax1.plot(target_ppx, P_ppx, "-", linewidth=3, color=colormap(2), label="Perple_X")

    if results_ml:
        ax1.plot(target_ml, P_ml, "-.", linewidth=3, color=colormap(1), label=f"{model}")

    if target == "rho":
        target_label = "Density"

    else:
        target_label = target

    ax1.set_xlabel(f"{target_label } ({target_unit})")
    ax1.set_ylabel("P (GPa)")
    ax1.set_xlim(target_min - (target_min * 0.05), target_max + (target_max * 0.05))
    ax1.set_xticks(np.linspace(target_min, target_max, num=4))

    if target in ["Vp", "Vs", "rho"]:
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
        plt.text(1 - text_margin_x, text_margin_y - (text_spacing_y * 0),
                 f"R$^2$: {r2_mean:.3f}", transform=plt.gca().transAxes,
                 fontsize=fontsize * 0.833, horizontalalignment="right",
                 verticalalignment="bottom")
        plt.text(1 - text_margin_x, text_margin_y - (text_spacing_y * 1),
                 f"RMSE: {rmse_mean:.3f}", transform=plt.gca().transAxes,
                 fontsize=fontsize * 0.833, horizontalalignment="right",
                 verticalalignment="bottom")

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize target array  !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_target_array(P, T, target_array, target, title, palette, color_discrete,
                           color_reverse, vmin, vmax, fig_dir, filename, figwidth=6.3,
                           figheight=4.725, fontsize=22):
    """
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
        num_colors = len(np.unique(target_array))

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

        im = ax.imshow(target_array, extent=[T.min(), T.max(), P.min(), P.max()],
                       aspect="auto", cmap=cmap, origin="lower", vmin=1, vmax=num_colors + 1)

        ax.set_xlabel("T (K)")
        ax.set_ylabel("P (GPa)")
        plt.colorbar(im, ax=ax, ticks=np.arange(1, num_colors + 1, num_colors // 4), label="")

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
            vmin=-np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))
            vmax=np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))

        else:
            vmin, vmax = vmin, vmax

            # Adjust vmin close to zero
            if vmin <= 1e-4:
                vmin = 0

            # Set melt fraction to 0–100 %
            if target == "melt_fraction":
                vmin, vmax = 0, 100

        # Set nan color
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color="white")

        # Plot as a raster using imshow
        fig, ax = plt.subplots()

        im = ax.imshow(target_array, extent=[T.min(), T.max(), P.min(), P.max()],
                       aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

        ax.set_xlabel("T (K)")
        ax.set_ylabel("P (GPa)")

        # Diverging colorbar
        if palette == "seismic":
            cbar = plt.colorbar(im, ax=ax, ticks=[vmin, 0, vmax], label="")

        # Continuous colorbar
        else:
            cbar = plt.colorbar(im, ax=ax, ticks=np.linspace(vmin, vmax, num=4), label="")

        # Set colorbar limits and number formatting
        if target == "rho":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vp":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
        elif target == "Vs":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "melt_fraction":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage_variance":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

    # Add title
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

def visualize_target_surf(P, T, target_array, target, title, palette, color_discrete,
                          color_reverse, vmin, vmax, fig_dir, filename, figwidth=6.3,
                          figheight=4.725, fontsize=22):
    """
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
        num_colors = len(np.unique(target_array))

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

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(T, P, target_array, cmap=cmap, vmin=1, vmax=num_colors + 1)

        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(title, y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")
        cbar = fig.colorbar(surf, ax=ax, ticks=np.arange(1, num_colors + 1, num_colors // 4),
                            label="", shrink=0.6)
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        cbar.ax.set_ylim(1, num_colors + 1)

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
            vmin=-np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))
            vmax=np.max(np.abs(target_array[np.logical_not(np.isnan(target_array))]))

        else:
            vmin, vmax = vmin, vmax

            # Adjust vmin close to zero
            if vmin <= 1e-4:
                vmin = 0

            # Set melt fraction to 0–100 %
            if target == "melt_fraction":
                vmin, vmax = 0, 100

        # Set nan color
        cmap = plt.cm.get_cmap(cmap)
        cmap.set_bad(color="white")

        # 3D surface
        fig = plt.figure(figsize=(figwidth, figheight), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(T, P, target_array, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel("T (K)", labelpad=18)
        ax.set_ylabel("P (GPa)", labelpad=18)
        ax.set_zlabel("")
        ax.set_zlim(vmin - (vmin * 0.05), vmax + (vmax * 0.05))
        plt.tick_params(axis="x", which="major")
        plt.tick_params(axis="y", which="major")
        plt.title(title, y=0.95)
        ax.view_init(20, -145)
        ax.set_box_aspect((1.5, 1.5, 1), zoom=1)
        ax.set_facecolor("white")

        # Diverging colorbar
        if palette == "seismic":
            cbar = fig.colorbar(surf, ax=ax, ticks=[vmin, 0, vmax], label="", shrink=0.6)

        # Continous colorbar
        else:
            cbar = fig.colorbar(surf, ax=ax, ticks=np.linspace(vmin, vmax, num=4),
                                label="", shrink=0.6)

        # Set z and colorbar limits and number formatting
        if target == "rho":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "Vp":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
        elif target == "Vs":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        elif target == "melt_fraction":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
        elif target == "assemblage_variance":
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

        cbar.ax.set_ylim(vmin, vmax)

    # Save fig
    plt.savefig(f"{fig_dir}/{filename}")

    # Close fig
    plt.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize training dataset !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_training_dataset(program, sample_id, res, dataset, targets, mask_geotherm,
                               palette, fig_dir, verbose):
    """
    """
    # Data assets dir
    data_dir = "assets/dir"

    # Model output dir
    out_dir = "runs"

    # Get training dataset results
    if program == "magemin":
        # Get MAGEMin results
        results = read_gfem_results(program, sample_id, dataset, res, verbose)
        program_title = "MAGEMin"

    elif program == "perplex":
        # Get perplex results
        results = read_gfem_results(program, sample_id, dataset, res, verbose)
        program_title = "Perple_X"

    else:
        raise ValueError("program argument must be MAGEMin or Perple_X!")

    # Get PT values
    P, T = results["P"], results["T"]

    # Get target array
    target_array = create_target_array(results, targets, mask_geotherm)

    for i, target in enumerate(targets):
        # Reshape targets into square array
        square_target = target_array[:, i].reshape(res+1, res+1)

        # Use discrete colorscale
        if target in ["assemblage", "assemblage_variance"]:
            color_discrete = True

        else:
            color_discrete = False

        # Reverse color scale
        if palette in ["grey"]:
            if target in ["assemblage_variance"]:
                color_reverse = True

            else:
                color_reverse = False

        else:
            if target in ["assemblage_variance"]:
                color_reverse = False

            else:
                color_reverse = True

        # Set colorbar limits for better comparisons
        if not color_discrete:
            vmin=np.min(square_target[np.logical_not(np.isnan(square_target))])
            vmax=np.max(square_target[np.logical_not(np.isnan(square_target))])

        else:
            num_colors = len(np.unique(square_target))

            vmin = 1
            vmax = num_colors + 1

        # Rename target
        target_rename = target.replace('_', '-')

        # Print filepath
        if verbose >= 1:
            print(f"Saving figure: {program}-{sample_id}-{dataset}-{target_rename}.png")

        # Plot targets
        visualize_target_array(P, T, square_target, target, program_title, palette,
                               color_discrete, color_reverse, vmin, vmax, fig_dir,
                               f"{program}-{sample_id}-{dataset}-{target_rename}.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize training dataset diff !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_training_dataset_diff(sample_id, res, dataset, targets, mask_geotherm, palette,
                                    fig_dir, verbose):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Model output dir
    out_dir = "runs"

    # Get MAGEMin results
    results_mgm = process_magemin_results(sample_id, dataset, res, verbose)

    # Get perplex results
    results_ppx = process_perplex_results(sample_id, dataset, res, verbose)

    # Get PT values
    P_mgm, T_mgm = results_mgm["P"], results_mgm["T"]
    P_ppx, T_ppx = results_ppx["P"], results_ppx["T"]

    # Get target arrays
    target_array_mgm = create_target_array(results_mgm, targets, mask_geotherm)
    target_array_ppx = create_target_array(results_ppx, targets, mask_geotherm)

    for i, target in enumerate(targets):
        # Reshape targets into square array
        square_array_mgm = target_array_mgm[:, i].reshape(res+1, res+1)
        square_array_ppx = target_array_ppx[:, i].reshape(res+1, res+1)

        # Use discrete colorscale
        if target in ["assemblage", "assemblage_variance"]:
            color_discrete = True

        else:
            color_discrete = False

        # Reverse color scale
        if palette in ["grey"]:
            if target in ["assemblage_variance"]:
                color_reverse = True

            else:
                color_reverse = False

        else:
            if target in ["assemblage_variance"]:
                color_reverse = False

            else:
                color_reverse = True

        # Set colorbar limits for better comparisons
        if not color_discrete:
            vmin_mgm=np.min(square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
            vmax_mgm=np.max(square_array_mgm[np.logical_not(np.isnan(square_array_mgm))])
            vmin_ppx=np.min(square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])
            vmax_ppx=np.max(square_array_ppx[np.logical_not(np.isnan(square_array_ppx))])

            vmin = min(vmin_mgm, vmin_ppx)
            vmax = max(vmax_mgm, vmax_ppx)

        else:
            num_colors_mgm = len(np.unique(square_array_mgm))
            num_colors_ppx = len(np.unique(square_array_ppx))

            vmin = 1
            vmax = max(num_colors_mgm, num_colors_ppx) + 1

        if not color_discrete:
            # Define a filter to ignore the specific warning
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")

            # Create nan mask
            mask = ~np.isnan(square_array_mgm) & ~np.isnan(square_array_ppx)

            # Compute normalized diff
            diff = square_array_mgm - square_array_ppx

            # Add nans to match original target arrays
            diff[~mask] = np.nan

            # Rename target
            target_rename = target.replace('_', '-')

            # Print filepath
            if verbose >= 1:
                print(f"Saving figure: diff-{sample_id}-{dataset}-{target_rename}.png")

            # Plot target array normalized diff mgm-ppx
            visualize_target_array(P_ppx, T_ppx, diff, target, "Residuals", "seismic",
                                   color_discrete, False, vmin, vmax, fig_dir,
                                   f"diff-{sample_id}-{dataset}-{target_rename}.png")

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
            if target == "rho":
                # Print filepath
                if verbose >= 1:
                    print(f"Saving figure: prem-{sample_id}-{dataset}-{target_rename}.png")

                visualize_prem(target, "g/cm$^3$", results_mgm, results_ppx,
                               geotherm_threshold=geotherm_threshold,
                               title="PREM Comparison", fig_dir=fig_dir,
                               filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

            if target in ["Vp", "Vs"]:
                # Print filepath
                if verbose >= 1:
                    print(f"Saving figure: prem-{sample_id}-{dataset}-{target_rename}.png")

                visualize_prem(target, "km/s", results_mgm, results_ppx,
                               geotherm_threshold=geotherm_threshold,
                               title="PREM Comparison", fig_dir=fig_dir,
                               filename=f"prem-{sample_id}-{dataset}-{target_rename}.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize cv results !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_cv_results(program, model, model_label, model_label_full, targets,
                         features_array, targets_array, scaler_X, scaler_y, X_scaled,
                         y_scaled, cv_info, seed, palette, vmin, vmax, fig_dir, filename,
                         figwidth, figheight, fontsize):
    """
    """
    # Get number of features and targets
    n_features = features_array.shape[-1]
    n_targets = targets_array.shape[-1]

    # Get length of features and targets
    W = int(np.sqrt(features_array.shape[0]))

    # Train model for plotting
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled,
                                                        test_size=0.2, random_state=seed)

    # Train ML model
    model.fit(X_train, y_train)

    # Scale features
    X = features_array
    X_scaled = scaler_X.fit_transform(X)

    # Make predictions
    pred_scaled = model.predict(X_scaled)

    # Inverse transform predictions
    pred_original = scaler_y.inverse_transform(pred_scaled)

    # Reshape arrays into squares for visualization
    features_array = features_array.reshape(W, W, n_features)
    targets_array = targets_array.reshape(W, W, n_targets)
    pred_array_original = pred_original.reshape(W, W, n_targets)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "0.9"
    plt.rcParams["legend.loc"] = "upper left"
    plt.rcParams["legend.fontsize"] = "small"
    plt.rcParams["legend.frameon"] = "False"
    plt.rcParams["axes.facecolor"] = "0.9"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["figure.autolayout"] = "True"
    plt.rcParams["figure.constrained_layout.use"] = "True"
    plt.rcParams["figure.dpi"] = 330
    plt.rcParams["savefig.bbox"] = "tight"

    # Colormap
    cmap = plt.cm.get_cmap("bone_r")
    cmap.set_bad(color="white")

    for i, target in enumerate(targets):
        # Create nan mask for validation set targets
        mask = np.isnan(targets_array[:, :, i])

        # Match nans between validation set predictions and original targets
        pred_array_original[:, :, i][mask] = np.nan

        # Compute normalized diff
        diff = targets_array[:,:,i] - pred_array_original[:,:,i]

        # Make nans consistent
        diff[mask] = np.nan

        # Plot training data distribution and ML model predictions
        colormap = plt.cm.get_cmap("tab10")

        # Reverse color scale
        if palette in ["grey"]:
            color_reverse = False

        else:
            color_reverse = True

        # Plot target array 2d
        visualize_target_array(features_array[:,:,0], features_array[:,:,1],
                               targets_array[:,:,i], target, f"{program}", palette, False,
                               color_reverse, vmin[i], vmax[i], fig_dir,
                               f"{filename}-{target.replace('_', '-')}-targets.png")

        # Plot target array 3d
        visualize_target_surf(features_array[:,:,0], features_array[:,:,1],
                              targets_array[:,:,i], target, f"{program}", palette, False,
                              color_reverse, vmin[i], vmax[i], fig_dir,
                              f"{filename}-{target.replace('_', '-')}-targets-surf.png")

        # Plot ML model predictions array 2d
        visualize_target_array(features_array[:,:,0], features_array[:,:,1],
                               pred_array_original[:,:,i], target, f"{model_label_full}",
                               palette, False, color_reverse, vmin[i], vmax[i], fig_dir,
                               f"{filename}-{target.replace('_', '-')}-predictions.png")

        # Plot ML model predictions array 3d
        visualize_target_surf(features_array[:,:,0], features_array[:,:,1],
                              pred_array_original[:,:,i], target, f"{model_label_full}",
                              palette, False, color_reverse, vmin[i], vmax[i], fig_dir,
                              f"{filename}-{target.replace('_', '-')}-surf.png")

        # Plot PT normalized diff targets vs. ML model predictions 2d
        visualize_target_array(features_array[:,:,0], features_array[:,:,1], diff, target,
                               "Residuals", "seismic", False, False, vmin[i], vmax[i],
                               fig_dir, f"{filename}-{target.replace('_', '-')}-diff.png")

        # Plot PT normalized diff targets vs. ML model predictions 3d
        visualize_target_surf(features_array[:,:,0], features_array[:,:,1], diff, target,
                              "Residuals", "seismic", False, False, vmin[i], vmax[i],
                              fig_dir, f"{filename}-{target.replace('_', '-')}-diff-surf.png")

        # Reshape results and transform units for MAGEMin
        if program == "magemin":
            results_mgm = {"P": features_array[:, :, 0].ravel().tolist(),
                           "T": features_array[:, :, 1].ravel().tolist(),
                           target: targets_array[:, :, i].ravel().tolist()}

            results_ppx = None

        # Reshape results and transform units for Perple_X
        if program == "perplex":
            results_ppx = {"P": features_array[:, :, 0].ravel().tolist(),
                           "T": features_array[:, :, 1].ravel().tolist(),
                           target: targets_array[:, :, i].ravel().tolist()}

            results_mgm = None

        # Reshape results and transform units for ML model
        results_rocml = {"P": features_array[:, :, 0].ravel().tolist(),
                         "T": features_array[:, :, 1].ravel().tolist(),
                         target: pred_array_original[:, :, i].ravel().tolist()}

        # Get relevant metrics for PREM plot
        rmse, r2 = cv_info[f"rmse_val_mean_{target}"], cv_info[f"r2_val_mean_{target}"]
        metrics = [rmse[0], r2[0]]

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

        # Plot PREM comparisons
        if target == "rho":
            visualize_prem(target, "g/cm$^3$", results_mgm, results_ppx, results_rocml,
                           f"{model_label}", geotherm_threshold, metrics,
                           title=f"{model_label_full}", fig_dir=fig_dir,
                           filename=f"{filename}-{target.replace('_', '-')}-prem.png")

        if target in ["Vp", "Vs"]:
            visualize_prem(target, "km/s", results_mgm, results_ppx, results_rocml,
                           f"{model_label}", geotherm_threshold, metrics,
                           title=f"{model_label_full}", fig_dir=fig_dir,
                           filename=f"{filename}-{target.replace('_', '-')}-prem.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# visualize rocml performance !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def visualize_rocml_performance(sample_id, target, res, fig_dir, filename, fontsize=22,
                                figwidth=6.3, figheight=4.725):
    """
    """
    # Data assets dir
    data_dir = "assets/data"

    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Read regression data
    data = pd.read_csv(f"{data_dir}/rocml-performance.csv")
    data = data[data["sample"] == sample_id]

    # Summarize data
    numeric_columns = data.select_dtypes(include=[float, int]).columns
    summary_df = data.groupby("model")[numeric_columns].mean().reset_index()

    # Get MAGEMin and Perple_X benchmark times
    benchmark_times = pd.read_csv(f"{data_dir}/benchmark-efficiency.csv")

    filtered_times = benchmark_times[(benchmark_times["sample"] == sample_id) &
                                     (benchmark_times["size"] == res**2)]

    time_mgm = np.mean(
        filtered_times[filtered_times["program"] == "magemin"]["time"].values /
        filtered_times[filtered_times["program"] == "magemin"]["size"].values *
        1000
    )
    time_ppx = np.mean(
        filtered_times[filtered_times["program"] == "perplex"]["time"].values /
        filtered_times[filtered_times["program"] == "perplex"]["size"].values *
        1000
    )

    # Define the metrics to plot
    metrics = ["training_time_mean", "inference_time_mean",
               f"rmse_test_mean_{target}", f"rmse_val_mean_{target}"]
    metric_names = ["Training Efficiency", "Prediction Efficiency",
                    "Training Error", "Validation Error"]

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
    colormap = plt.cm.get_cmap("tab10")
    models = ["KN", "DT", "RF", "NN1", "NN2", "NN3"]

    # Create a dictionary to map each model to a specific color
    color_mapping = {"KN": colormap(2), "DT": colormap(0), "RF": colormap(4),
                     "NN1": colormap(1), "NN2": colormap(3), "NN3": colormap(5)}

    # Get the corresponding colors for each model
    colors = [color_mapping[model] for model in models]

    # Define units
    if target == "rho":
        unit = "(kg/m$^3$)"
    elif target in ["Vp", "Vs"]:
        unit = "(km/s)"
    elif target == "melt_fraction":
        unit = "(%)"
    else:
        unit = ""

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

        # Show MAGEMin and Perple_X compute times
        if metric == "inference_time_mean":
            mgm_line = plt.axhline(time_mgm, color="black", linestyle="-", label="MAGEMin")
            ppx_line = plt.axhline(time_ppx, color="black", linestyle="--", label="Perple_X")

        # Plot the bars for each program
        bars = plt.bar(x_positions * bar_width, summary_df.loc[order][metric],
                       edgecolor="black", width=bar_width,
                       color=[color_mapping[model] for model in models_order],
                       label=models_order if i == 1 else "")

        plt.gca().set_xticks([])
        plt.gca().set_xticklabels([])

        # Plot titles
        if metric == "training_time_mean":
            plt.title(f"{metric_names[i]}")
            plt.ylabel("Elapsed Time (ms)")
            plt.yscale("log")

        elif metric == "inference_time_mean":
            plt.title(f"{metric_names[i]}")
            plt.ylabel("Elapsed Time (ms)")
            plt.yscale("log")
            handles = [mgm_line, ppx_line]
            labels = [handle.get_label() for handle in handles]
            legend = plt.legend(fontsize="x-small")
            legend.set_bbox_to_anchor((0.44, 0.89))

        elif metric == f"rmse_test_mean_{target}":
            # Calculate limits
            max_error = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_std_{target}"].values * 2,
                summary_df.loc[order][f"rmse_val_std_{target}"].values * 2
            ]))

            max_mean = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_mean_{target}"].values,
                summary_df.loc[order][f"rmse_val_mean_{target}"].values
            ]))

            vmax = max_mean + max_error + ((max_mean + max_error) * 0.05)

            plt.errorbar(x_positions * bar_width,
                         summary_df.loc[order][f"rmse_test_mean_{target}"],
                         yerr=summary_df.loc[order][f"rmse_test_std_{target}"] * 2,
                         fmt="none", capsize=5, color="black", linewidth=2)

            plt.title(f"{metric_names[i]}")
            plt.ylabel(f"RMSE {unit}")
            plt.ylim(0, vmax)

            if target != "melt_fraction":
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

            else:
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

        elif metric == f"rmse_val_mean_{target}":
            # Calculate limits
            max_error = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_std_{target}"].values * 2,
                summary_df.loc[order][f"rmse_val_std_{target}"].values * 2
            ]))

            max_mean = np.max(np.concatenate([
                summary_df.loc[order][f"rmse_test_mean_{target}"].values,
                summary_df.loc[order][f"rmse_val_mean_{target}"].values
            ]))

            vmax = max_mean + max_error + ((max_mean + max_error) * 0.05)

            plt.errorbar(x_positions * bar_width,
                         summary_df.loc[order][f"rmse_val_mean_{target}"],
                         yerr=summary_df.loc[order][f"rmse_val_std_{target}"] * 2,
                         fmt="none", capsize=5, color="black", linewidth=2)

            plt.title(f"{metric_names[i]}")
            plt.ylabel(f"RMSE {unit}")
            plt.ylim(0, vmax)

            if target != "melt_fraction":
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

            else:
                plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

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