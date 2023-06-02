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
        print(f"Directory \"log\" created successfully.")
        print("===========================================================")
    else:
        print(f"Directory \"log\" already exists ...")
        print("===========================================================")
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
        pressure_range (list): A list containing the start, end, and step values for the pressure range.
        temperature_range (list): A list containing the start, end, and step values for the temperature range.
        composition (list): A list of floats for the bulk composition with the following oxides
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
        for result in results:
            print(f"Point {result["Point"]}:")
            print(f"Status: {result["Status"]}")
            print(f"P: {result["P"]}")
            print(f"T: {result["T"]}")
            print(f"Gibbs: {result["Gibbs"]}")
            print(f"BrNorm: {result["BrNorm"]}")
            print(f"Gamma: {result["Gamma"]}")
            print(f"Vp: {result["Vp"]}")
            print(f"Vs: {result["Vs"]}")
            print(f"Entropy: {result["Entropy"]}")
            print(f"Stable Solutions: {result["StableSolutions"]}")
            print(f"Stable Fractions: {result["StableFractions"]}")
            print(f"Density: {result["Density"]}")
            print(f"Compositional Var: {result["CompositionalVar"]}")
            print(f"EM Fractions: {result["EMFractions"]}")
            print(f"EM List: {result["EMList"]}")
            print(f"Liquid Fraction: {result["LiquidFraction"]}")
            print(f"Density of Full Assemblage: {result["DensityOfFullAssemblage"]}")
            print(f"Density of Liquid: {result["DensityOfLiquid"]}")
            print(f"Density of Solid: {result["DensityOfSolid"]}")
            print(f"Density of Mixture: {result["DensityOfMixture"]}")
            print()

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
                em_frac = data[4+n_xeos::2]
                em = out[5+n_xeos::2]

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
            "Gamma": gamma,
            "Vp": vp,
            "Vs": vs,
            "Entropy": entropy,
            "StableSolutions": stable_solutions,
            "StableFractions": stable_fractions,
            "Density": density,
            "CompositionalVar": compositional_var,
            "EMFractions": em_fractions,
            "EMList": em_list,
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
        list: A list of processed results from the files.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Example:
        results = process_files("output/", "_pseudosection*.txt")
        for result in results:
            print(f"Point {result["Point"]}:")
            print(f"Status: {result["Status"]}")
            print(f"P: {result["P"]}")
            print(f"T: {result["T"]}")
            print(f"Gibbs: {result["Gibbs"]}")
            print(f"BrNorm: {result["BrNorm"]}")
            print(f"Gamma: {result["Gamma"]}")
            print(f"Vp: {result["Vp"]}")
            print(f"Vs: {result["Vs"]}")
            print(f"Entropy: {result["Entropy"]}")
            print(f"Stable Solutions: {result["StableSolutions"]}")
            print(f"Stable Fractions: {result["StableFractions"]}")
            print(f"Density: {result["Density"]}")
            print(f"Compositional Var: {result["CompositionalVar"]}")
            print(f"EM Fractions: {result["EMFractions"]}")
            print(f"EM List: {result["EMList"]}")
            print(f"Liquid Fraction: {result["LiquidFraction"]}")
            print(f"Density of Full Assemblage: {result["DensityOfFullAssemblage"]}")
            print(f"Density of Liquid: {result["DensityOfLiquid"]}")
            print(f"Density of Solid: {result["DensityOfSolid"]}")
            print(f"Density of Mixture: {result["DensityOfMixture"]}")
            print()

    """
    results = []

    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(root, filename)
            file_results = read_pseudosection_data_MAGEMin(filepath)
            results.extend(file_results)

    return results

# Run MAGEMin and visualize
def run_MAGEMin(MAGEMin_program_path, MAGEMin_input_path, MAGEMin_output_directory,
                MAGEMin_output_pattern, MAGEMin_mode, MAGEMin_parallel,
                MAGEMin_parallel_nprocs, MAGEMin_pressure_range,
                MAGEMin_temperature_range, MAGEMin_composition, MAGEMin_param,
                MAGEMin_plot=True, MAGEMin_plot_fname="MAGEMin-fig-1.png", verbose = False):
    # Look for previous MAGEMin output
    MAGEMin_output_file_list = glob.glob(MAGEMin_output_directory + "/" + MAGEMin_output_pattern)
    if len(MAGEMin_output_file_list) == 0:
        # Run your code here if no files matching the pattern are found
        print("No MAGEMin output files found ...")
        print("Creating MAGEMin input file ...")
        print("===========================================================")
        print(f"P range: from {MAGEMin_pressure_range[0]}–{MAGEMin_pressure_range[1]} kbar")
        print(f"T range: from {MAGEMin_temperature_range[0]}–{MAGEMin_temperature_range[1]} C")
        print(
            f"Composition [wt%]: \n"
            f"SiO2({MAGEMin_composition[0]}) Al2O3({MAGEMin_composition[1]}) CaO({MAGEMin_composition[2]}) MgO({MAGEMin_composition[3]}) FeOt({MAGEMin_composition[4]}) K2O({MAGEMin_composition[5]}) Na2O({MAGEMin_composition[6]}) TiO2({MAGEMin_composition[7]}) O({MAGEMin_composition[8]}) Cr2O3({MAGEMin_composition[9]}) H2O({MAGEMin_composition[10]})"
        )
        print("===========================================================")
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
        print(f"Running MAGEMin in the shell at {n_MAGEMin_pt_points} pt points ...")
        print(f"{MAGEMin_mpi_exec}")
        print("===========================================================")
        shell_process = subprocess.run(
            MAGEMin_mpi_exec, shell=True, capture_output=True, text=True
        )
        # Print terminal output
        if shell_process.stdout is not None:
            print("Terminal stdout:")
            print(shell_process.stdout)
            print("===========================================================")
        if shell_process.stderr is not None:
            print("Terminal stderr:")
            print(shell_process.stderr)
            print("===========================================================")
        # Process MAGEMin output files
        MAGEMin_output = process_MAGEMin_files(
            MAGEMin_output_directory,
            MAGEMin_output_pattern
        )
        print(f"MAGEMin output [P (kbar), T (˚C), {MAGEMin_param}: ")
        if(verbose == True): print(MAGEMin_output)
        print("===========================================================")
    else:
        # If previous output exists just visualize
        print("MAGEMin output found ...")
        print(f"Plotting {MAGEMin_param} ...")
        print("===========================================================")
        # Process MAGEMin output files
        MAGEMin_output = process_MAGEMin_files(
            MAGEMin_output_directory,
            MAGEMin_output_pattern
        )
    if MAGEMin_plot == True:
        # Create a directory if it does not exist
        if not os.path.exists("figs"):
            os.makedirs("figs")
            print(f"Directory \"figs\" created successfully.")
            print("===========================================================")
        else:
            print(f"Directory \"figs\" already exists ...")
            print("===========================================================")