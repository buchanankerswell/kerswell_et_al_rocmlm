# Load packages
import os
import re
import sys
import glob
import time
import shutil
import zipfile
import fnmatch
import itertools
import subprocess
import numpy as np
import pandas as pd
from git import Repo
import urllib.request
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    print(f"Downloading data from:\n{url}")
    urllib.request.urlretrieve(url, "data.zip")

    # Extract the contents of the zip file
    print(f"Extracting the contents of the zip file to {destination}/ ...")
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall(destination)

    # Remove the zip file
    os.remove("data.zip")

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

# Normalize components
def normalize_concentrations(concentrations, components="all"):
    """
    Normalize the concentrations for a subset of components.

    Args:
        concentrations (list): List of concentrations representing the components.
        components (list): List of components to normalize the concentrations for.

    Returns:
        list: Normalized concentrations for each component in the same order.

    Raises:
        ValueError: If the input concentrations list doesn't have exactly 11 components.

    """
    if components == "all":
        return concentrations
    component_list = [
        "SiO2", "Al2O3", "CaO", "MgO", "FeOt", "K2O",
        "Na2O", "TiO2", "O(Fe2O3)", "Cr2O3", "H2O"
    ]
    if len(concentrations) != 11:
        error_message = (
            f"The input concentrations list must have exactly 11 components ...\n" +
            f"{component_list}"
        )
        raise ValueError(error_message)
    if len(concentrations) != 11:
        raise ValueError("The input concentrations list must have exactly 11 components ...")
    subset_concentrations = [
        c if comp in components else 0.01 for c, comp in zip(concentrations, component_list)
    ]
    total_subset_concentration = sum([c for c in subset_concentrations if c != 0.01])
    normalized_concentrations = []
    for c, comp in zip(concentrations, component_list):
        if comp in components:
            normalized_concentration = (
                (c / total_subset_concentration)*100 if c != 0.01 else 0.01
            )
        else:
            normalized_concentration = 0.01
        normalized_concentrations.append(normalized_concentration)
    return normalized_concentrations

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
            f"--n_points={n_points} --sys_in={comp_type} --db={database}"
        )

    # Run MAGEMin
    if(verbose == True):
        exec = exec + " --Verb=1"
        shell_process = subprocess.run(exec, shell=True, text=True)
    else:
        shell_process = subprocess.run(exec, shell=True, text=True)

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
def process_MAGEMin_files(run_name):
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
        encoded_assemblage = unique_assemblages[tuple(sorted(phase_assemblage))]
        encoded_assemblages.append(encoded_assemblage)

    return encoded_assemblages, unique_assemblages

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

    Example:
        P = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
        T = [25, 25, 25, 50, 50, 50]
        parameter_values = [1, 2, 3, 4, 5, 6]
        grid = create_PT_grid(P, T, parameter_values)
        print(grid)

    Output:
        [[1. 2. 3.]
         [4. 5. 6.]]
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
def plot_pseudosection(P, T, grid, parameter, title=None, palette="blues", filename=None):
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
    if not os.path.exists("figs"):
        os.makedirs("figs", exist_ok=True)
    if (parameter == "StableSolutions"):
        # Create a custom colormap with dynamically determined colors
        num_colors = len(np.unique(grid))
        # Color palette
        if palette == "viridis":
            pal = plt.cm.get_cmap("viridis", num_colors)
        elif palette == "grey":
            pal = plt.cm.get_cmap("Greys", num_colors)
        elif palette not in ["viridis", "grey"]:
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
            vmin=0,
            vmax=num_colors-1
        )
        ax.set_xlabel("T (˚C)")
        ax.set_ylabel("P (kbar)")
        plt.colorbar(im, ax=ax, ticks=np.arange(num_colors), label=parameter)
        fig.tight_layout()
        if title:
            plt.title(title)
    else:
        # Color palette
        if palette == "viridis":
            cmap = "viridis"
        elif palette == "grey":
            cmap = "Greys"
        elif palette not in ["viridis", "grey"]:
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
        plt.savefig("figs/" + filename, bbox_inches="tight", dpi=330)
    else:
        # Print plot
        plt.show()

# Plot Harker diagram with density contours using seaborn
def plot_harker_diagram(data, x_oxide, y_oxide, filename=None):
    """
    Plot Harker diagrams with density contours using seaborn.

    Parameters:
        data (pandas.DataFrame): The geochemical data to be plotted.
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
    # Create figs dir
    if not os.path.exists("figs"):
        os.makedirs("figs", exist_ok=True)

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
            hue="COMPOSITION",
            hue_order=["ultramafic", "mafic"],
            fill=True,
            ax=ax
        )

        # X-Y scatter (Harker diagram)
        sns.scatterplot(
            data=data,
            x=x_oxide.upper(),
            y=y.upper(),
            hue="COMPOSITION",
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
    n_mafic = len(data[data["COMPOSITION"] == "mafic"])
    n_ulmafic = len(data[data["COMPOSITION"] == "ultramafic"])

    fig.suptitle(f"Harker diagrams for {n_mafic} mafic and {n_ulmafic} ultramafic samples")
    plt.tight_layout()

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig("figs/" + filename, bbox_inches="tight", dpi=330)
    else:
        # Print plot
        plt.show()