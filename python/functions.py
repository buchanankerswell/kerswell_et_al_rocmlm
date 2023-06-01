# Load packages
import os
import re
import sys
import glob
import logging
import itertools
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Logging
def setup_logging(log_file_path):
    """
    Set up logging configuration to log messages to a file and stdout.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        None. Redirects print statements to the log file and stdout.

    """
    # Create a directory if it doesn't exist
    if not os.path.exists("log"):
        os.makedirs("log")
        print(f"Directory 'log' created successfully.")
        print("===========================================================")
    else:
        print(f"Directory 'log' already exists ...")
        print("===========================================================")
    # Create a custom log formatter without the prefix
    class LogFormatter(logging.Formatter):
        def format(self, record):
            return record.message
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='')
    log = logging.getLogger()
    # Set the formatter for the log file
    file_handler = logging.FileHandler(log_file_path)
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
            if message.rstrip() != '':
                self.logger.log(self.level, message.rstrip())

        def flush(self):
            pass
    sys.stdout = LogWriter(log)

# Count lines in a file
def count_lines(file_path):
    """
    Count the number of lines in a file.

    This function opens the specified file and counts the number of lines present in the file.

    Args:
        file_path (str): The path to the file.

    Returns:
        int: The number of lines in the file.

    """
    line_count = 0
    with open(file_path, "r") as file:
        for line in file:
            line_count += 1
    return line_count

# Create MAGEmin input
def create_magemin_input(mode, pressure_range, temperature_range, composition, file_path = "test-input.dat", print_input = False):
    """
    Creates an input string for MAGEmin.

    Args:
        mode (str): The mode value for MAGEmin.
        pressure_range (list): A list containing the start, end, and step values for the pressure range.
        temperature_range (list): A list containing the start, end, and step values for the temperature range.
        composition (list): A list of floats for the bulk composition with the following oxides
                            [SiO2, Al2O3, CaO, MgO, FeOt, K2O, Na2O, TiO2, O, Cr2O3, H2O]
    Returns:
        (file): Writes the input string for MAGEmin to file_path.
    """
    magemin_input = ""
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
        magemin_input += (
            f"{mode} {p} {t} {composition[0]} {composition[1]} {composition[2]} {composition[3]}"
            f"{composition[4]} {composition[5]} {composition[6]} {composition[7]} {composition[8]}"
            f"{composition[9]} {composition[10]}\n"
        )
    with open(file_path, "w") as f:
        f.write(magemin_input)
    if print_input == True:
        return magemin_input

# Get PT from MAGEmin input
def extract_magemin_input_pt(data_file):
    """
    Given a MAGEmin output file, extracts the input P and T values
    used for the calculation.

    Args:
        data_file (file): A _matlab_output.txt file from the MAGEmin program.

    Returns:
        list: Returns a list with the two values in the order (P, T).
              If the values cannot be found, returns None.
    """
    with open(data_file, "r") as file:
        found = False
        results = []
        for line in file:
            if "===" in line:
                found = True
                continue
            if found and any(c.isdigit() or c in ".-+e" for c in line):
                results.append([float(num) for num in re.findall(r"[-+]?\d*\.\d+", line)])
                found = False
        if results:
            return results
    return None

# Get "Stable assemblage" output table
def extract_magemin_assemblage_table(data_file):
    """
    Given a MAGEmin output file,
    extracts the stable mineral assemblage table as a string.

    Args:
        data_file (file): A _matlab_output.txt file from the MAGEmin program.

    Returns:
        list: Returns a list of strings representing each table found.
              If no table is found, returns an empty list.
    """
    with open(data_file, "r") as file:
        lines = file.readlines()
        start_line = None
        end_line = None
        tables = []
        for i, line in enumerate(lines):
            if line.startswith("Stable mineral assemblage:"):
                start_line = i + 1
            elif start_line is not None and line.strip() == "":
                end_line = i - 1
            if start_line is not None and end_line is not None:
                table_lines = lines[start_line:end_line]
                table_lines = [line.strip() for line in table_lines]
                tables.append("\n".join(table_lines))
                start_line = None
                end_line = None
        if tables:
            return tables
        else:
            return None

# Calculate weighted average for one parameter calculated by MAGEmin
def calculate_weighted_average(data_table, param):
    """
    Given a string representation of a table and the name of a column,
    calculates the weighted average of the values in that column.

    Args:
        data_table (str): A string representation of a table extracted
                          from a MAGEmin output file.

    Returns:
        list: Returns the weighted average as a float.
    """
    rows = data_table.split("\n")
    header = rows[0].split()
    data = [list(map(float, row.split()[1:])) for row in rows[1:] if row.strip()]
    column_index = header.index(param)-1
    total_weighted_value = 0
    total_weight = 0
    for row in data:
        phase_weight = row[1]
        phase_value = row[column_index]
        total_weighted_value += phase_weight * phase_value
        total_weight += phase_weight
    weighted_average = total_weighted_value / total_weight
    return weighted_average

# Extract PT and param data from MAGEmin output
def get_magemin_param_at_input_pt_point(data_file, param):
    """
    Given a MAGEmin output file and the name of a parameter,
    extracts the input P and T values used for the calculation,
    calculates the weighted average of the parameter for the stable mineral assemblage,
    and returns a list with the three values in the order (P, T, weighted average of the parameter).
    If any of the required data cannot be found, returns None.

    Args:
        data_file (file): A single _matlab_output.txt file from the MAGEmin program.
        param (str): A parameter from the "Stable assemblage" table
                     G[J], V_molar[cm3/mol], V_partial[cm3], Cp[kJ/K], Rho[kg/m3],
                     Alpha[1/K], Entropy[J/K], Enthalpy[J], BulkMod[GPa],
                     ShearMod[GPa], Vp[km/s], Vs[km/s]

    Returns:
        list: Returns a list with the three values in the order (P, T, weighted average of the parameter).
    """
    magemin_input_pt = extract_magemin_input_pt(data_file)
    assemblage_table = extract_magemin_assemblage_table(data_file)
    whole_rock_avg = []
    for table in assemblage_table:
        whole_rock_avg.append([calculate_weighted_average(table, param)])
    pt_param_points = [pt + avg for pt, avg in zip(magemin_input_pt, whole_rock_avg)]
    return pt_param_points

# Process all MAGEmin output files in a directory
def process_magemin_output_files(directory, pattern, param):
    """
    Process MAGEmin output files and extract a specific parameter.

    This function searches for files in the given directory that match the provided pattern,
    reads each file, and extracts the specified parameter using the 'get_magemin_param_at_input_pt_point'
    function. The extracted parameter values are then flattened and stacked into a NumPy array.

    Args:
        directory (str): The directory path where the MAGEmin output files are located.
        pattern (str): The pattern to match the filenames of the MAGEmin output files.
        param (str): The parameter to extract from the MAGEmin output files.

    Returns:
        numpy.ndarray: A NumPy array containing the extracted parameter values.

    """
    file_paths = glob.glob(directory + pattern)
    processed_magemin_output = []
    for data_file in file_paths:
        processed_magemin_output.append(get_magemin_param_at_input_pt_point(data_file, param))
    processed_magemin_output_flat = [item for sublist in processed_magemin_output for item in sublist]
    processed_magemin_output_array = np.vstack(processed_magemin_output_flat)
    return processed_magemin_output_array

# Run MAGEmin and visualize
def run_magemin(magemin_program_path, magemin_input_path, magemin_output_directory,
                magemin_output_pattern, magemin_mode, magemin_parallel,
                magemin_parallel_nprocs, magemin_pressure_range,
                magemin_temperature_range, magemin_composition, magemin_param,
                magemin_plot=True, magemin_plot_fname="magemin-fig-1.png", verbose = False):
    # Look for previous MAGEmin output
    magemin_output_file_list = glob.glob(magemin_output_directory + "/" + magemin_output_pattern)
    if len(magemin_output_file_list) == 0:
        # Run your code here if no files matching the pattern are found
        print("No MAGEmin output files found ...")
        print("Creating MAGEmin input file ...")
        print("===========================================================")
        print(f"P range: from {magemin_pressure_range[0]}–{magemin_pressure_range[1]} kbar")
        print(f"T range: from {magemin_temperature_range[0]}–{magemin_temperature_range[1]} C")
        print(
            f"Composition [wt%]: \n"
            f"SiO2({magemin_composition[0]}) Al2O3({magemin_composition[1]}) CaO({magemin_composition[2]}) MgO({magemin_composition[3]}) FeOt({magemin_composition[4]}) K2O({magemin_composition[5]}) Na2O({magemin_composition[6]}) TiO2({magemin_composition[7]}) O({magemin_composition[8]}) Cr2O3({magemin_composition[9]}) H2O({magemin_composition[10]})"
        )
        print("===========================================================")
        # Write MAGEmin input file
        create_magemin_input(
            magemin_mode, magemin_pressure_range, magemin_temperature_range,
            magemin_composition, magemin_input_path, True
        )
        # Count number of pt points to model with MAGEmin
        n_magemin_pt_points = count_lines(magemin_input_path)
        # Execute MAGEmin
        if magemin_parallel == True:
            magemin_mpi_exec = (
                f"mpirun -np {magemin_parallel_nprocs} {magemin_program_path}MAGEMin "
                f"--File={magemin_input_path} --n_points={n_magemin_pt_points} "
                f"--db=ig --test=0 --out_matlab=1"
            )
        else:
            magemin_mpi_exec = (
                f"{magemin_program_path}MAGEMin --File={magemin_input_path} "
                f"--n_points={n_magemin_pt_points} --db=ig --test=0 --out_matlab=1 "
            )
        # Call MAGEmin in the terminal
        print(f"Running MAGEmin in the shell at {n_magemin_pt_points} pt points ...")
        print(f"{magemin_mpi_exec}")
        print("===========================================================")
        shell_process = subprocess.run(
            magemin_mpi_exec, shell=True, capture_output=True, text=True
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
        # Process MAGEmin output files
        magemin_output = process_magemin_output_files(
            magemin_output_directory, magemin_output_pattern, magemin_param
        )
        print(f"MAGEmin output [P (kbar), T (˚C), {magemin_param}: ")
        if(verbose == True): print(magemin_output)
        print("===========================================================")
    else:
        # If previous output exists just visualize
        print("MAGEmin output found ...")
        print(f"Plotting {magemin_param} ...")
        print("===========================================================")
        # Process MAGEmin output files
        magemin_output = process_magemin_output_files(
            magemin_output_directory, magemin_output_pattern, magemin_param
        )
    if magemin_plot == True:
        # Create a directory if it doesn't exist
        if not os.path.exists("figs"):
            os.makedirs("figs")
            print(f"Directory 'figs' created successfully.")
            print("===========================================================")
        else:
            print(f"Directory 'figs' already exists ...")
            print("===========================================================")
        fig, ax = plt.subplots(figsize=(4.25, 3.5))
        plt.scatter(
            magemin_output[:, 1],
            magemin_output[:, 0],
            c=magemin_output[:, 2],
            marker="s"
        )
        colorbar = plt.colorbar()
        colorbar.set_label(magemin_param)
        plt.xlabel('Temperature (˚C)')
        plt.ylabel('Pressure (kbar)')
        plt.savefig("figs/" + magemin_plot_fname, dpi=330, bbox_inches="tight")