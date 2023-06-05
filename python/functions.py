# Load packages
import os
import re
import sys
import glob
import fnmatch
import logging
import itertools
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Logging
def setup_logging(filename):
    """
    Set up logging configuration to log messages to a file and stdout.

    Args:
        filename (str): Path to the log file.

    Returns:
        None. Redirects print statements to the log file and stdout.

    """
    # Create a directory if it does not exist
    if not os.path.exists("log"):
        os.makedirs("log")
    # Create a custom log formatter without the prefix
    class LogFormatter(logging.Formatter):
        def format(self, record):
            return record.message
    logging.basicConfig(filename=filename, level=logging.INFO, format="")
    log = logging.getLogger()
    # Set the formatter for the log file
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    log.addHandler(file_handler)
    # Set the formatter for stdout logging without the prefix
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(LogFormatter())
    log.addHandler(stdout_handler)
    # Redirect print statements to the log file and stdout
    class LogWriter:
        def __init__(self, logger, level=logging.INFO):
            self.logger = logger
            self.level = level

        def write(self, message):
            if message.rstrip() != "":
                self.logger.log(self.level, message.rstrip())

        def flush(self):
            pass
    sys.stdout = LogWriter(log)

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
        mode, pressure_range,
        temperature_range,
        composition,
        filename = "test-input.dat",
        print_input = False):
    """
    Creates an input string for MAGEMin.

    Args:
        mode (str): The mode value for MAGEMin.
        pressure_range (list): Start, end, and step values for the pressure range.
        temperature_range (list): Start, end, and step values for the temperature range.
        composition (list): Bulk composition with the following oxides
            [SiO2, Al2O3, CaO, MgO, FeOt, K2O, Na2O, TiO2, O, Cr2O3, H2O]
    Returns:
        (file): Writes the input string for MAGEMin to filename.
    """
    MAGEMin_input = ""
    pressure_values = np.arange(
        float(pressure_range[0]),
        float(pressure_range[1]) + 1,
        float(pressure_range[2])
    )
    temperature_values = np.arange(
        float(temperature_range[0]),
        float(temperature_range[1]) + 1,
        float(temperature_range[2])
    )
    combinations = list(itertools.product(pressure_values, temperature_values))
    for p, t in combinations:
        MAGEMin_input += (
            f"{mode} {p} {t} {composition[0]} {composition[1]} {composition[2]} {composition[3]}"
            f"{composition[4]} {composition[5]} {composition[6]} {composition[7]} {composition[8]}"
            f"{composition[9]} {composition[10]}\n"
        )
    with open(filename, "w") as f:
        f.write(MAGEMin_input)
    if print_input == True:
        return MAGEMin_input

# Read pseudosection info from MAGEMin output file
def read_pseudosection_data_MAGEMin(filename):
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
        results = read_pseudosection_data_MAGEMin("_pseudosection.txt")
    """
    with open(filename, "r") as file:
        lines = file.readlines()

    # Skip the comment line
    lines = lines[1:]

    results = []
    i_point = 0
    while lines:
        # Read line with P/T and Gamma
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

        # Compute average Density of full assemblage
        density_total = sum(frac * dens for frac, dens in zip(stable_fractions, density))

        # Compute Density of liq
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
def process_MAGEMin_files(directory, pattern):
    """
    Process multiple files in a directory based on a filename pattern.

    Args:
        directory (str): The directory path where the files are located.
        pattern (str): The filename pattern to match the files.

    Returns:
        dict: A single dictionary with merged key-value pairs from all files.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    results = []

    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(root, filename)
            file_results = read_pseudosection_data_MAGEMin(filepath)
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
def plot_pseudosection_results(results, parameter, filename=None):
    """
    Plot the results of a pseudosection calculation.

    Args:
        results (dict): Dictionary containing the results of a pseudosection calculation.
        parameter (str): The parameter to plot. If "StableSolutions", phase assemblages
        will be plotted, otherwise, the specified parameter values will be plotted.
        filename (str, optional): If provided, the plot will be saved to the specified file.

    Returns:
        None
    """
    P = results["P"]
    T = results["T"]
    if (parameter == "StableSolutions"):
        # Encode unique phase assemblages
        encoded, unique = encode_phases(results[parameter])
        legend_elements = []
        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(T, P, c=encoded, cmap="viridis", marker=",")
        ax.set_xlabel("T (˚C)")
        ax.set_ylabel("P (kbar)")
        # Legend
        for assemblage_tuple, encoded_number in unique.items():
            assemblage_name = ", ".join(assemblage_tuple)
            legend_elements.append(
                ax.scatter(
                    [],
                    [],
                    marker="o",
                    color=scatter.cmap(scatter.norm(encoded_number)),
                    label=assemblage_name
                )
            )
#        legend = ax.legend(
#            handles=legend_elements,
#            title="Phase Assemblage",
#            bbox_to_anchor=(0.5, -0.15),
#            loc="upper center",
#            ncol=2
#        )
        plt.subplots_adjust(bottom=0.15)
#        legend.get_frame().set_facecolor("white")
        fig.tight_layout(rect=[0, 0, 1, 1])
    else:
        parameter_values = results[parameter]
        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(T, P, c=parameter_values, cmap="viridis", marker=",")
        ax.set_xlabel("T (˚C)")
        ax.set_ylabel("P (kbar)")
        plt.colorbar(scatter, ax=ax, label=parameter)

    # Print plot
    plt.show()

    # Save the plot to a file if a filename is provided
    if filename:
        plt.savefig(filename)

# Run MAGEMin and visualize
def run_MAGEMin(MAGEMin_program_path, MAGEMin_input_path, MAGEMin_output_directory,
                MAGEMin_output_pattern, MAGEMin_mode, MAGEMin_parallel,
                MAGEMin_parallel_nprocs, MAGEMin_pressure_range,
                MAGEMin_temperature_range, MAGEMin_composition, MAGEMin_param,
                MAGEMin_plot=True, MAGEMin_plot_fname="MAGEMin-fig-1.png", verbose = False):
    # Look for previous MAGEMin output
    MAGEMin_output_file_list = glob.glob(
        MAGEMin_output_directory + "/" + MAGEMin_output_pattern
    )
    if len(MAGEMin_output_file_list) == 0:
        # Run your code here if no files matching the pattern are found
        print("No MAGEMin output files found ...")
        print("Creating MAGEMin input file ...")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"P range: from {MAGEMin_pressure_range[0]}–{MAGEMin_pressure_range[1]} kbar")
        print(f"T range: from {MAGEMin_temperature_range[0]}–{MAGEMin_temperature_range[1]} C")
        print(
            f"Composition [wt%]: \n"
            f"SiO2({MAGEMin_composition[0]})\nAl2O3({MAGEMin_composition[1]})\nCaO({MAGEMin_composition[2]})\nMgO({MAGEMin_composition[3]})\nFeOt({MAGEMin_composition[4]})\nK2O({MAGEMin_composition[5]})\nNa2O({MAGEMin_composition[6]})\nTiO2({MAGEMin_composition[7]})\nO({MAGEMin_composition[8]})\nCr2O3({MAGEMin_composition[9]})\nH2O({MAGEMin_composition[10]})"
        )
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # Write MAGEMin input file
        create_MAGEMin_input(
            MAGEMin_mode, MAGEMin_pressure_range, MAGEMin_temperature_range,
            MAGEMin_composition, MAGEMin_input_path, True
        )
        # Count number of pt points to model with MAGEMin
        n_MAGEMin_pt_points = count_lines(MAGEMin_input_path)
        # Execute MAGEMin
        if MAGEMin_parallel == True:
            MAGEMin_mpi_exec = (
                f"mpirun -np {MAGEMin_parallel_nprocs} {MAGEMin_program_path}MAGEMin "
                f"--File={MAGEMin_input_path} --n_points={n_MAGEMin_pt_points} "
                f"--db=ig --test=0 --out_matlab=1"
            )
        else:
            MAGEMin_mpi_exec = (
                f"{MAGEMin_program_path}MAGEMin --File={MAGEMin_input_path} "
                f"--n_points={n_MAGEMin_pt_points} --db=ig --test=0 --out_matlab=1 "
            )
        # Call MAGEMin in the terminal
        print(f"Running MAGEMin in the shell for {n_MAGEMin_pt_points} pt points ...")
        print(f"{MAGEMin_mpi_exec}")
        shell_process = subprocess.run(
            MAGEMin_mpi_exec, shell=True, capture_output=True, text=True
        )
        # Print terminal output
        if(verbose == True):
            if shell_process.stdout is not None:
                print("Terminal stdout:")
                print(shell_process.stdout)
                print("===========================================================")
            if shell_process.stderr is not None:
                print("Terminal stderr:")
                print(shell_process.stderr)
                print("===========================================================")
        # Process MAGEMin output files
        print("Processing MAGEMin output ...")
        results = process_MAGEMin_files(
            MAGEMin_output_directory,
            MAGEMin_output_pattern
        )
        if(verbose == True):
            print(results)
    else:
        # If previous output exists just visualize
        print("MAGEMin output found ...")
        print("Processing MAGEMin output ...")
        # Process MAGEMin output files
        results = process_MAGEMin_files(
            MAGEMin_output_directory,
            MAGEMin_output_pattern
        )
        if(verbose == True):
            print(results)
    if MAGEMin_plot == True:
        print(f"Plotting {MAGEMin_param} ...")
        # Create a directory if it does not exist
        if not os.path.exists("figs"):
            os.makedirs("figs")
        print(f"Saving pseudosection plot to figs/{MAGEMin_plot_fname}")
        plot_pseudosection_results(results, MAGEMin_param, "figs" + MAGEMin_plot_fname)