#!/opt/homebrew/Caskroom/miniforge/base/bin/python

# Load packages
import glob
import datetime

from functions import (
    setup_logging, count_lines, create_MAGEMin_input,
    read_pseudosection_data_MAGEMin, process_MAGEMin_files, run_MAGEMin
)

# Logging
current_date = datetime.date.today()
log_file = f"log/log-{current_date}"
setup_logging(log_file)

# Run MAGEMin
run_MAGEMin(
    MAGEMin_program_path = "MAGEMin/",
    MAGEMin_input_path = "MAGEMin/test-input.dat",
    MAGEMin_output_directory = "output/",
    MAGEMin_output_pattern = "_pseudosection*.txt",
    MAGEMin_mode = 0,
    MAGEMin_parallel = True,
    MAGEMin_parallel_nprocs = 6,
    MAGEMin_pressure_range = [10, 100, 1],
    MAGEMin_temperature_range = [200, 2000, 10],
    MAGEMin_composition = [41.49, 1.57, 4.824, 52.56, 5.88, 0.01, 0.25, 0.1, 0.1, 0.0, 0.0],
    MAGEMin_param = "StableSolutions",
    MAGEMin_plot=True,
    MAGEMin_plot_fname="test.png",
    verbose=False
)

print("Done!")