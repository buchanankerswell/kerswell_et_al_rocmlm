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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Read conda packages from yaml
def get_conda_packages(conda_file):
    try:
        with open(conda_file, 'r') as file:
            conda_data = yaml.safe_load(file)
        return conda_data.get('dependencies', [])
    except (IOError, yaml.YAMLError) as e:
        print(f"Error reading Conda file: {e}")
        return []

# Read makefile variables
def read_makefile_variable(makefile, variable):
    try:
        with open(makefile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip().startswith(variable):
                    return line.split('=')[1].strip()
    except IOError as e:
        print(f"Error reading Makefile: {e}")
    return None

# Print session info for logging
def print_session_info(conda_file=None, makefile=None):
    print("Session info:")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Print Python version
    python_version = sys.version_info
    version_string = '.'.join(map(str, python_version))
    print(f"Python Version: {version_string}")

    # Print package versions
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Loaded packages:")
    if conda_file:
        conda_packages = get_conda_packages(conda_file)
        for package in conda_packages:
            if isinstance(package, str) and package != "python":
                package_name = package.split('=')[0]
                try:
                    version = pkg_resources.get_distribution(package_name).version
                    print(f"{package_name} Version: {version}")
                except pkg_resources.DistributionNotFound:
                    print(f"{package_name} not found.")
    else:
        print("No Conda file provided.")

    # Print operating system information
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    os_info = platform.platform()
    print(f"Operating System: {os_info}")

    # Print random seed
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    if makefile:
        seed = read_makefile_variable(makefile, 'SEED')
        if seed:
            print(f"Random Seed (from Makefile): {seed}")
        else:
            print("SEED variable not found in Makefile.")
    else:
        print("No Makefile provided.")

# Check non-matching strings
def check_non_matching_strings(list1, list2):
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
        argparse.ArgumentTypeError: If the input string is not a valid list of three numbers.

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
                f"Invalid list: {arg} ...\nIt must contain a valid list of strings."
            )
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid list: {arg} ...\nIt must contain a valid list of strings."
        )

# Parse arguments for build-database.py
def parse_arguments_build_db():
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
        required=True
    )
    parser.add_argument(
        "--Pmax",
        type=int,
        help="Specify the Pmax argument ...",
        required=True
    )
    parser.add_argument(
        "--Pres",
        type=int,
        help="Specify the Pres argument ...",
        required=True
    )
    parser.add_argument(
        "--Tmin",
        type=int,
        help="Specify the Tmin argument ...",
        required=True
    )
    parser.add_argument(
        "--Tmax",
        type=int,
        help="Specify the Tmax argument ...",
        required=True
    )
    parser.add_argument(
        "--Tres",
        type=int,
        help="Specify the Tres argument ...",
        required=True
    )
    parser.add_argument(
        "--comp",
        type=parse_list_of_numbers,
        help="Specify the comp argument ...",
        required=True
    )
    parser.add_argument(
        "--frac",
        type=str,
        help="Specify the frac argument ...",
        required=True
    )
    parser.add_argument(
        "--sampleid",
        type=str,
        help="Specify the sampleid argument ...",
        required=True
    )
    parser.add_argument(
        "--normox",
        type=parse_list_of_strings,
        help="Specify the normox argument ...",
        required=True
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Specify the source argument ...",
        required=True
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Specify the strategy argument ...",
        required=True
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Specify the n argument ...",
        required=True
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Specify the k argument ...",
        required=True
    )
    parser.add_argument(
        "--parallel",
        type=str,
        help="Specify the parallel argument ...",
        required=True
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        help="Specify the nprocs argument ...",
        required=True
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Specify the seed argument ...",
        required=True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Specify the outdir argument ...",
        required=True
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Convert the parallel argument to a boolean value based on the string value
    parallel = args.parallel.lower() == "true" if args.parallel else False

    # Assign the modified parallel value back to the args object
    args.parallel = parallel

    return args

# Parse arguments for visualize-database.py
def parse_arguments_visualize_db():
    """
    Parse the command-line arguments for the visualize-database.py script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.

    Raises:
        argparse.ArgumentTypeError: If any of the required arguments are missing.

    """
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the command-line arguments
    parser.add_argument(
        "--sampleid",
        type=str,
        help="Specify the sampleid argument ...",
        required = True
    )
    parser.add_argument(
        "--params",
        type=parse_list_of_strings,
        help="Specify the params argument ...",
        required = True
    )
    parser.add_argument(
        "--figox",
        type=parse_list_of_strings,
        help="Specify the figox argument ...",
        required = True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Specify the outdir argument ...",
        required = True
    )
    parser.add_argument(
        "--figdir",
        type=str,
        help="Specify the figdir argument ...",
        required = True
    )

    # Parse the command-line arguments
    return parser.parse_args()

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
    print(f"Downloading assets from:\n{url}")
    urllib.request.urlretrieve(url, "assets.zip")

    # Extract the contents of the zip file
    print(f"Extracting the contents of the zip file to {destination}/ ...")
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
def batch_sample_for_MAGEMin(datafile, batch_size=1, k=0):
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
def get_benchmark_sample_for_MAGEMin(datafile, sample_id):
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
    subset_df = df[df['NAME'] == sample_id]

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
def random_sample_for_MAGEMin(datafile, n=1, seed=None):
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
        ValueError: If the input sample list doesn't have exactly 11 components.

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
                (c / total_subset_concentration)*100 if c != 0.01 else 0.01
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

    Example:
        download_github_submodule("https://github.com/username/repo.git", "submodule_dir")
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
def create_MAGEMin_input(
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
    MAGEMin_input = ""
    pressure_values = np.arange(
        float(P_range[0]),
        float(P_range[1]) + 1,
        float(P_range[2])
    )
    temperature_values = np.arange(
        float(T_range[0]),
        float(T_range[1]) + 1,
        float(T_range[2])
    )

    # Expand PT vectors into grid
    combinations = list(itertools.product(pressure_values, temperature_values))
    for p, t in combinations:
        MAGEMin_input += (
            f"{mode} {p} {t} {composition[0]} {composition[1]} {composition[2]} "
            f"{composition[3]} {composition[4]} {composition[5]} {composition[6]} "
            f"{composition[7]} {composition[8]} {composition[9]} {composition[10]}\n"
        )

    # Write input file
    with open(f"{directory}/{run_name}.dat", "w") as f:
        f.write(MAGEMin_input)

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
def run_MAGEMin(
        program_path=None,
        run_name="test",
        comp_type="mol",
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
        verbose (bool, optional): Determines whether to print verbose output. Default is True.

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
def read_MAGEMin_pseudosection_data(filename):
    """
    Read and process pseudosection data from a file.

    Args:
        filename (str): The path to the file containing the pseudosection data.

    Returns:
        dict: A dictionary containing the processed pseudosection data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If there is an error in parsing the data.

    Example:
        results = read_MAGEMin_pseudosection_data("_pseudosection.txt")
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
                comp_var = data[4:4+n_xeos]
                em = out[4+n_xeos::2]
                em_frac = data[5+n_xeos::2]

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
def process_MAGEMin_grid(run_name, out_dir="runs"):
    """
    Process multiple MAGEMin output files in a directory based on a filename pattern.

    Args:
        run_name (str): The name of the MAGEMin run.

    Returns:
        dict: A single dictionary with merged key-value pairs from all files.

    Raises:
        FileNotFoundError: If the "runs" directory or the specific MAGEMin run
        directory does not exist.
    """
    # Check for MAGEMin output files
    if not os.path.exists(f"{os.getcwd()}/{out_dir}"):
        sys.exit("No MAGEMin output files to process ...")
    if not os.path.exists(f"{os.getcwd()}/{out_dir}/{run_name}"):
        sys.exit("No MAGEMin output files to process ...")

    # Get filenames directory for files
    directory = f"{os.getcwd()}/{out_dir}/{run_name}"
    pattern = f"_{run_name}*.txt"

    results = []

    # Process files
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(root, filename)
            file_results = read_MAGEMin_pseudosection_data(filepath)
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
def encode_phases(phases):
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
    phase_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            phases = line.split('-')[1].strip().split()
            cleaned_phases = [phase.split('(')[0].lower() for phase in phases]
            phase_dict[line_number] = cleaned_phases
    return phase_dict

# Transform results into 2D numpy array across PT space
def create_PT_grid(P, T, parameter_values):
    """
    Create a 2D NumPy array representing a grid of parameter values across PT space.

    Parameters:
        P (list or array-like): A 1D array or list of P values.
        T (list or array-like): A 1D array or list of T values.
        parameter_values (list or array-like): A 1D array or list of parameter values.

    Returns:
        grid (numpy.ndarray): A 2D NumPy array representing the grid of parameter values.
            The grid is created by reshaping the parameter values based on unique
            P and T values.
            Missing values in the grid are represented by NaN.

    """
    # Convert P, T, and parameter_values to arrays
    P = np.array(P)
    T = np.array(T)
    parameter_values = np.array(parameter_values)

    # Get unique P and T values
    unique_P = np.unique(P)
    unique_T = np.unique(T)

    # Determine the grid dimensions
    rows = len(unique_P)
    cols = len(unique_T)

    # Create an empty grid to store the reshaped parameter_values
    grid = np.empty((rows, cols))

    # Reshape the parameter_values to match the grid dimensions
    for i, p in enumerate(unique_P):
        for j, t in enumerate(unique_T):
            index = np.where((P == p) & (T == t))[0]
            if len(index) > 0:
                grid[i, j] = parameter_values[index[0]]

    return grid

# Plot MAGEMin pseudosection
def plot_pseudosection(
        P,
        T,
        grid,
        parameter,
        title=None,
        palette="blues",
        color_discrete=False,
        color_reverse=False,
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
        filename (str, optional): If provided, the plot will be saved to the specified file.

    Returns:
        None
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
    if color_discrete:
        # Create a custom colormap with dynamically determined colors
        num_colors = len(np.unique(grid))
        # Color palette
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
            vmin=1,
            vmax=num_colors+1
        )
        ax.set_xlabel("T (˚C)")
        ax.set_ylabel("P (kbar)")
        plt.colorbar(im, ax=ax, ticks=np.arange(num_colors)+1, label=parameter)
        fig.tight_layout()
        if title:
            plt.title(title)
    else:
        # Color palette
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
            origin="lower"
        )
        ax.set_xlabel("T (˚C)")
        ax.set_ylabel("P (kbar)")
        plt.colorbar(im, ax=ax, label=f"{parameter}")
        fig.tight_layout()
        if title:
            plt.title(title)

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}", bbox_inches="tight", dpi=330)
    else:
        # Print plot
        plt.show()

    # Close device
    plt.close()


# Plot Harker diagram with density contours using seaborn
def plot_harker_diagram(
        datafile,
        x_oxide="SiO2",
        y_oxide="MgO",
        filename="earthchem-harker.png",
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Plot Harker diagrams with density contours using seaborn.

    Parameters:
        data (filepath): Path to the geochemical datafile in .csv format.
        x_oxide (str): The x-axis oxide for the Harker diagram.
        y_oxide (str or list): The y-axis oxide(s) for the Harker diagram.
        Can be a single oxide or a list of oxides.
        filename (str, optional): The filename to save the plot.
        If not provided, the plot will be displayed interactively.

    Returns:
        None

    Notes:
        - The function creates a directory named "figs" to save the plots.
        - The function uses seaborn's "dark" style, palette, and "talk" context for the plots.
        - The function creates a grid of subplots based on the number of y-oxides specified.
        - Density contours and scatter plots (Harker diagrams) are added to each subplot.
        - Legends are shown only for the first subplot.
        - The plot can be saved to a file if a filename is provided.
    """
    # Read geochemical data
    data = read_geochemical_data(datafile)
    data = data.rename(columns={"COMPOSITION": "Rock Type"})

    # Create figs dir
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    # Set plot style and settings
    sns.set_style("dark")
    sns.set_context("talk")
    plt.rcParams["legend.facecolor"] = "white"

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
    fig_width = 6*num_cols
    fig_height = 5*num_rows

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

        ax.set_facecolor("0.5")
        ax.set_xlabel(f"{x_oxide} (wt%)")
        ax.set_ylabel(f"{y} (wt%)")

        # Show legend only for the first subplot
        if i > 0:
            ax.get_legend().remove()

    # Remove empty subplots
    if num_plots < len(axes):
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])

    # Count the number of mafic and ultramafic samples
    n_mafic = len(data[data["Rock Type"] == "mafic"])
    n_ulmafic = len(data[data["Rock Type"] == "ultramafic"])

    fig.suptitle(f"Harker diagrams for {n_mafic} mafic and {n_ulmafic} ultramafic samples")
    plt.tight_layout()

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}", bbox_inches="tight", dpi=330)
    else:
        # Print plot
        plt.show()

# Visualize benchmark comp times
def visualize_benchmark_comp_times(
        datafile,
        filename="benchmark-comp-times.png",
        fig_dir=f"{os.getcwd()}/figs"):
    """
    """
    # Check for figs directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)
    # Read data
    data = pd.read_csv(datafile)

    # Set plot style and settings
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "0.5"

    # Create a dictionary to map methods to marker styles and line styles
    method_styles = {
        "magemin": {"marker": "o", "linestyle": "-"},
        "perplex": {"marker": "s", "linestyle": "--"}
    }

    # Create a dictionary to map samples to colors
    sample_colors = {
        "DMM": "blue",
        "NMORB1": "green",
        "NMORB2": "red",
        "PUM": "orange",
        "RE46": "purple"
    }

    # Get max resolution
    data["maxres"] = data[["tres", "pres"]].max(axis=1)

    # Group the data by method and sample
    grouped_data = data.groupby(["method", "sample"])

    # Plot the data
    for group, group_data in grouped_data:
        method_val, sample_val = group
        marker_style = method_styles[method_val]["marker"]
        linestyle_val = method_styles[method_val]["linestyle"]
        color_val = sample_colors[sample_val]

        # Filter out rows with missing time values
        filtered_data = group_data.dropna(subset=["time"])

        # Extract x and y values
        x = filtered_data["maxres"]
        y = filtered_data["time"]

        # Plot the data points and connect them with lines
        plt.plot(
            x, y,
            marker=marker_style,
            color=color_val,
            linestyle=linestyle_val,
            label=f"{method_val} {sample_val}"
        )

    # Set labels and title
    plt.xlabel("PT Grid Resolution")
    plt.ylabel("Time (s)")
    plt.title("Gibbs Minimization Efficiency")
    plt.xticks([8, 16, 32, 64, 128])
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(f"{fig_dir}/{filename}", bbox_inches="tight", dpi=330)
    else:
        # Print plot
        plt.show()