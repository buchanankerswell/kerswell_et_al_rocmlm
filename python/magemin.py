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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import ListedColormap

# Layout benchmark plots horizontally
def combine_plots_horizontally(
        image1_path,
        image2_path,
        output_path,
        caption1,
        caption2,
        font_size=100,
        caption_margin=25,
        dpi=330):
    """
    Combine benchmark plots horizontally and add captions in the upper left corner.

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
    combined_image = Image.new('RGB', (max_width*2, max_height), (255, 255, 255))

    # Set the DPI metadata
    combined_image.info['dpi'] = (dpi, dpi)

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

# Layout benchmark plots vertically
def combine_plots_vertically(image1_path, image2_path, output_path):
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
    combined_image = Image.new('RGB', (max_width, combined_height), (255, 255, 255))

    # Paste the first image on the top
    combined_image.paste(image1, (0, 0))

    # Paste the second image below the first
    combined_image.paste(image2, (0, image1.height))

    # Save the combined image
    combined_image.save(output_path)
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
    """
    Print session information for logging.

    Parameters:
        conda_file (str, optional): The path to the Conda YAML file. Defaults to None.
        makefile (str, optional): The path to the Makefile. Defaults to None.

    """
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
        "--colormap",
        type=str,
        help="Specify the colormap argument ...",
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
def plot_MAD(
        P,
        T,
        grid,
        parameter,
        T_unit = "K",
        P_unit = "GPa",
        title=None,
        palette="blues",
        color_discrete=False,
        color_reverse=False,
        vmin=None,
        vmax=None,
        figwidth=6.3,
        figheight=4.725,
        fontsize=20,
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
            vmax=num_colors+1
        )
        ax.set_xlabel(f"T ({T_unit})")
        ax.set_ylabel(f"P ({P_unit})")
        plt.colorbar(im, ax=ax, ticks=np.arange(1, num_colors+1, num_colors // 6), label="")
        if title:
            plt.title(title)
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
        if parameter in ["Vp", "Vs", "LiquidFraction", "DensityOfFullAssemblage"]:
            cbar = plt.colorbar(im, ax=ax, label="")
            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        else:
            plt.colorbar(im, ax=ax, label="")
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


# Plot Harker diagram with density contours using seaborn
def plot_harker_diagram(
        datafile,
        x_oxide="SiO2",
        y_oxide="MgO",
        fontsize=14,
        filename="earthchem-harker.png",
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
    fig_width = 3.15*num_cols
    fig_height = 3.15*num_rows

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
def visualize_benchmark_comp_times(
        datafile,
        palette="tab10",
        fontsize=14,
        figwidth=6.3,
        figheight=3.54,
        filename="benchmark-times.png",
        fig_dir=f"{os.getcwd()}/figs"):
    """
    Visualize benchmark computation times.

    Parameters:
        datafile (str): Path to the benchmark data file in .csv format.
        palette (str, optional): The color palette for sample colors. Default is "tab10".
        fontsize (int, optional): The font size for text elements in the plot. Default is 12.
        filename (str, optional): The filename to save the plot. If not provided,
            the plot will be displayed interactively. Default is "benchmark-times.png".
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

    # Arrange data by grid resolution and sample
    data.sort_values(by=["grid", "sample"], inplace=True)

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
    colormap = cm.get_cmap(palette)
    sample_colors = {
        "DMM": colormap(0),
        "NMORB": colormap(1),
        "PUM": colormap(3),
        "RE46": colormap(4)
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
        mgm_data = group_data.dropna(subset=["mgm"])
        mgm_x = np.sqrt(mgm_data["grid"])
        mgm_y = mgm_data["mgm"]

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

    # Plot the data for Perple_X lines
    for group, group_data in grouped_data:
        sample_val = group[0]
        color_val = sample_colors[sample_val]

        # Filter out rows with missing time values for ppx column
        ppx_data = group_data.dropna(subset=["ppx"])
        ppx_x = np.sqrt(ppx_data["grid"])
        ppx_y = ppx_data["ppx"]

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
    plt.xlabel("PT Grid Resolution")
    plt.ylabel("Time (s)")
    plt.title("GFEM Efficiency")
    plt.xticks([16, 32, 64, 128])

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
        fontsize=14,
        figwidth=6.3,
        figheight=3.54,
        filename="madnn-training-pt-range.png",
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

    # Define P range
    P_min, P_max = 1.0, 28.0

    # Olivine --> Ringwoodite Clapeyron slopes
    references_410 = {
        "Akaogi89": [0.001, 0.002],
        "Katsura89": [0.0025],
        "Morishima94": [0.0034, 0.0038]
    }

    # Ringwoodite --> Bridgmanite + Ferropericlase Clapeyron slopes
    references_660 = {
        "Ito82": [-0.002],
        "Ito89 & Hirose02": [-0.0028],
        "Ito90": [-0.002, -0.006],
        "Katsura03": [-0.0004, -0.002],
        "Akaogi07": [-0.0024, -0.0028]
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
    colormap = cm.get_cmap(palette)
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
            label = f"[410] {ref}" if j == 0 else None
            plt.plot(
                T[(T >= 1200) & (T <= 2273)],
                line[(T >= 1200) & (T <= 2273)],
                color=color,
                label=label
            )
            if label not in label_color_mapping:
                label_color_mapping[label] = color

    # Ringwoodite --> Bridgmanite + Ferropericlase
    for j, (ref, ref_lines) in enumerate(zip(references_660.keys(), lines_660)):
        color = colors[j+i+1 % len(colors)]
        for j, line in enumerate(ref_lines):
            label = f"[660] {ref}" if j == 0 else None
            plt.plot(
                T[(T >= 1200) & (T <= 2273)],
                line[(T >= 1200) & (T <= 2273)],
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
        where=(T >= 773) & (T <= 2273),
        color="gray",
        alpha=0.2
    )

    # Calculate mantle geotherms
    geotherm1 = (T - 273) / (0.5 * 35)
    geotherm2 = (T - 1200 - 273) / (0.5 * 35)
    geotherm3 = (T - 1500 - 273) / (0.5 * 35)

    # Plot mantle geotherms
    plt.plot(T, geotherm1, ":", color="black")
    plt.plot(T, geotherm2, "--", color="black")
    plt.plot(T, geotherm3, "-.", color="black")

    # Geotherm legend handles
    geotherm1_handle = mlines.Line2D(
        [], [], linestyle=":", color="black", label="Geotherm 1"
    )
    geotherm2_handle = mlines.Line2D(
        [], [], linestyle="--", color="black", label="Geotherm 2"
    )
    geotherm3_handle = mlines.Line2D(
        [], [], linestyle="-.", color="black", label="Geotherm 3"
    )

    # Phase boundaries legend handles
    ref_line_handles = [
        mlines.Line2D([], [], color=color, label=label) \
        for label, color in label_color_mapping.items() if label
    ]

    # Add geotherms to legend handles
    ref_line_handles.extend([geotherm1_handle, geotherm2_handle, geotherm3_handle])

    training_data_handle = mpatches.Patch(
        color="gray",
        alpha=0.2,
        label="Training Data Range"
    )

    labels_660.add("Training Data Range")
    label_color_mapping["Training Data Range"] = "gray"

    # Define the desired order of the legend items
    desired_order = [
        "[410] Akaogi89",
        "[410] Katsura89",
        "[410] Morishima94",
        "[660] Ito82",
        "[660] Ito89 & Hirose02",
        "[660] Ito90",
        "[660] Katsura03",
        "[660] Akaogi07",
        "Geotherm 1",
        "Geotherm 2",
        "Geotherm 3",
        "Training Data Range"
    ]

    # Sort the legend handles based on the desired order
    legend_handles = sorted(
        ref_line_handles + [training_data_handle],
        key=lambda x: desired_order.index(x.get_label())
    )

    plt.xlabel(f"Temperature ({T_unit})")
    plt.ylabel(f"Pressure ({P_unit})")
    plt.title("Mantle Transition Zones")
    plt.xlim(700, 2346)
    plt.ylim(0, 29)
    plt.yticks([0, 10, 20, 30])

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