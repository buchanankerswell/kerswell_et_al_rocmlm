# Load packages
import os
import re
import sys
import glob
import time
import shutil
import fnmatch
import itertools
import subprocess
import numpy as np
from git import Repo
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
        This function checks if the submodule directory already exists and deletes it.
        Then it clones the submodule and recursively clones its contents.

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

    This function opens the specified file and counts the number of lines present in the file.

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

# Move files from MAGEMin output dir
def cleanup_ouput_dir(run_name):
    """
    Move files from the MAGEMin output directory to a new directory based on the run name.
    Also moves the input data file into the new directory and removes the output directory.

    Args:
        run_name (str): The name of the run, used to create the new directory and rename
        the files.

    Returns:
        None

    Raises:
        None
    """
    # Create the directory based on the run_name
    directory="runs/" + run_name
    os.makedirs(directory, exist_ok=True)

    # Get a list of all files in the "output" directory matching the pattern
    files = os.listdir("output")
    matching_files = [file for file in files if file.startswith("_pseudosection")]

    # Rename and move the files
    for file in matching_files:
        new_filename = "_" + run_name + file[len("_pseudosection"):]
        new_filepath = os.path.join(directory, new_filename)
        old_filepath = os.path.join("output", file)
        shutil.move(old_filepath, new_filepath)

    # Move input data file into directory
    shutil.move(directory + ".dat", directory)

    # Remove the output directory
    shutil.rmtree("output")

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
def create_MAGEMin_input(P_range, T_range, composition, mode=0, run_name="test"):
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
    if not os.path.exists("runs"):
        os.makedirs("runs")
    # Print ranges and compositions
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
    with open("runs/" + run_name + ".dat", "w") as f:
        f.write(MAGEMin_input)

# Run MAGEMin
def run_MAGEMin(
        program_path=None,
        run_name="test",
        mode=0,
        comp_type="mol",
        database="ig",
        parallel=True,
        nprocs=os.cpu_count()-2,
        verbose=False):
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

    # Check for input MAGEMin input files
    if not os.path.exists("runs"):
        sys.exit("No MAGEMin input files to run ...")

    # Count number of pt points to model with MAGEMin
    input_path = "runs/" + run_name + ".dat"
    n_points = count_lines(input_path)

    # Execute MAGEMin in parallel with MPI
    if parallel == True:
        exec = (
            f"mpirun -np {nprocs} {program_path}MAGEMin --File={input_path} "
            f"--n_points={n_points} --sys_in={comp_type} --db={database}"
        )
    # Or execute MAGEMin in sequence
    else:
        exec = (
            f"{program_path}MAGEMin --File={input_path} "
            f"--n_points={n_points} --sys_in={comp_type} --db={database} "
        )

    # Run MAGEMin
    if(verbose == True):
        shell_process = subprocess.run(exec, shell=True, text=True)
    else:
        shell_process = subprocess.run(exec, shell=True)

    # Move output files and cleanup directory
    cleanup_ouput_dir(run_name)

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
                density_sol = sum(frac * dens for frac, dens in zip([stable_fractions[idx] for idx in ind_sol], [density[idx] for idx in ind_sol])) / sum([stable_fractions[idx] for idx in ind_sol])
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
def process_MAGEMin_files(run_name):
    """
    Process multiple MAGEMin output files in a directory based on a filename pattern.

    Args:
        run_name (str): The name of the MAGEMin run.

    Returns:
        dict: A single dictionary with merged key-value pairs from all files.

    Raises:
        FileNotFoundError: If the "runs" directory or the specific MAGEMin run directory does not exist.
    """
    # Check for input MAGEMin input files
    if not os.path.exists("runs"):
        sys.exit("No MAGEMin output files to process ...")
    if not os.path.exists("runs/" + run_name):
        sys.exit("No MAGEMin output files to process ...")

    # Get filenames directory for files
    directory = "runs/" + run_name
    pattern = "_" + run_name + "*.txt"

    results = []

    # Process files
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(root, filename)
            file_results = read_MAGEMin_pseudosection_data(filepath)
            results.extend(file_results)

    merged_results = merge_dictionaries(results)
    return merged_results

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
        encoded_assemblage = unique_assemblages[tuple(sorted(phase_assemblage))]
        encoded_assemblages.append(encoded_assemblage)

    return encoded_assemblages, unique_assemblages


# Plot MAGEMin pseudosection
def plot_pseudosection(results, parameter, filename=None):
    """
    Plot the results of a pseudosection calculation.

    Args:
        results (dict): Results of a pseudosection calculation.
        parameter (str): The parameter to plot. If "StableSolutions", phase assemblages
        will be plotted, otherwise, the specified parameter values will be plotted.
        filename (str, optional): If provided, the plot will be saved to the specified file.

    Returns:
        None
    """
    # Check for figs directory
    if not os.path.exists("figs"):
        os.makedirs("figs", exist_ok=True)
    P = results["P"]
    T = results["T"]
    if (parameter == "StableSolutions"):
        # Encode unique phase assemblages
        encoded, unique = encode_phases(results[parameter])

        # Convert P, T, and parameter_values to arrays
        P = np.array(P)
        T = np.array(T)
        parameter_values = np.array(encoded)

        # Normalize P and T between 0 and 1
        normalized_P = (P - P.min()) / (P.max() - P.min())
        normalized_T = (T - T.min()) / (T.max() - T.min())

        # Get unique normalized P and T values
        unique_normalized_P = np.unique(normalized_P)
        unique_normalized_T = np.unique(normalized_T)

        # Determine the grid dimensions
        rows = len(unique_normalized_P)
        cols = len(unique_normalized_T)

        # Create an empty grid to store the reshaped parameter_values
        grid = np.empty((rows, cols))

        # Reshape the parameter_values to match the grid dimensions
        for i, normalized_p in enumerate(unique_normalized_P):
            for j, normalized_t in enumerate(unique_normalized_T):
                index = np.where(
                    (normalized_P == normalized_p) &
                    (normalized_T == normalized_t)
                )[0]
                if len(index) > 0:
                    grid[i, j] = parameter_values[index[0]]

        # Create a custom colormap with dynamically determined colors
        num_colors = len(unique)

        # Viridis
        viridis_palette = plt.cm.get_cmap('viridis', num_colors)
        # Greyscale
        greyscale_palette = plt.cm.get_cmap('Greys', num_colors)
        # Descritize
        color_palette = greyscale_palette(np.linspace(0, 1, num_colors))
        cmap = ListedColormap(color_palette)

        # Plot as a raster using imshow
        fig, ax = plt.subplots()
        im = ax.imshow(
            grid,
            cmap=cmap,
            extent=[
                normalized_T.min(),
                normalized_T.max(),
                normalized_P.min(),
                normalized_P.max()
            ],
            origin='lower',
            vmin=0,
            vmax=num_colors-1
        )
        ax.set_xlabel("Normalized T")
        ax.set_ylabel("Normalized P")
        plt.colorbar(im, ax=ax, ticks=np.arange(num_colors), label=parameter)
        fig.tight_layout()
    else:
        parameter_values = results[parameter]
        # Convert P, T, and parameter_values to arrays
        P = np.array(P)
        T = np.array(T)
        parameter_values = np.array(parameter_values)

        # Normalize P and T between 0 and 1
        normalized_P = (P - P.min()) / (P.max() - P.min())
        normalized_T = (T - T.min()) / (T.max() - T.min())
        normalized_parameter_values = (
            (parameter_values - parameter_values.min()) /
            (parameter_values.max() - parameter_values.min())
        )


        # Get unique normalized P and T values
        unique_normalized_P = np.unique(normalized_P)
        unique_normalized_T = np.unique(normalized_T)

        # Determine the grid dimensions
        rows = len(unique_normalized_P)
        cols = len(unique_normalized_T)

        # Create an empty grid to store the reshaped parameter_values
        grid = np.empty((rows, cols))

        # Reshape the parameter_values to match the grid dimensions
        for i, normalized_p in enumerate(unique_normalized_P):
            for j, normalized_t in enumerate(unique_normalized_T):
                index = np.where(
                    (normalized_P == normalized_p) &
                    (normalized_T == normalized_t)
                )[0]
                if len(index) > 0:
                    grid[i, j] = normalized_parameter_values[index[0]]

        # Plot as a raster using imshow
        fig, ax = plt.subplots()
        im = ax.imshow(
            grid,
            cmap="Greys",
            extent=[
                normalized_T.min(),
                normalized_T.max(),
                normalized_P.min(),
                normalized_P.max()
            ],
            origin='lower'
        )
        ax.set_xlabel("Normalized T")
        ax.set_ylabel("Normalized P")
        plt.colorbar(im, ax=ax, label=f"Normalized {parameter}")
        fig.tight_layout()

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig("figs/" + filename, dpi=330)
    else:
        # Print plot
        plt.show()