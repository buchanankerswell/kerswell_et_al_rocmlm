#!/opt/homebrew/Caskroom/miniforge/base/bin/python

# Load packages
import glob
import datetime

from functions import (
    create_magemin_input,
    extract_magemin_input_pt,
    extract_magemin_assemblage_table,
    calculate_weighted_average,
    get_magemin_param_at_input_pt_point,
    process_magemin_output_files,
    count_lines, setup_logging,
    run_magemin
)

# Logging
current_date = datetime.date.today()
log_file = f"log/log-{current_date}"
setup_logging(log_file)

# MAGEmin paths
magemin_program_path = "MAGEMin/"
magemin_input_path = "MAGEMin/test-input.dat"
magemin_output_directory = "output/"
magemin_output_pattern = "_matlab_output*.txt"
magemin_output_file_list = glob.glob(magemin_output_directory + "/" + magemin_output_pattern)

# MAGEmin inputs
magemin_mode = 0
magemin_parallel = True
magemin_parallel_nprocs = 6
magemin_pressure_range = [30, 100, 1]
magemin_temperature_range = [300, 1000, 10]
magemin_composition = [41.49, 1.57, 4.824, 52.56, 5.88, 0.01, 0.25, 0.1, 0.1, 0.0, 0.0]

# Param to visualize
magemin_param = "Rho[kg/m3]"

# Run MAGEmin
run_magemin(
    magemin_program_path,
    magemin_input_path,
    magemin_output_directory,
    magemin_output_pattern,
    magemin_mode,
    magemin_parallel,
    magemin_parallel_nprocs,
    magemin_pressure_range,
    magemin_temperature_range,
    magemin_composition,
    magemin_param,
    magemin_plot=True,
    magemin_plot_fname="test.png",
    verbose=True
)

print("Done!")