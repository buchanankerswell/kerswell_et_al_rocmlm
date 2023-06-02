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

# MAGEMin paths
MAGEMin_program_path = "MAGEMin/"
MAGEMin_input_path = "MAGEMin/test-input.dat"
MAGEMin_output_directory = "output/"
MAGEMin_output_pattern = "_pseudosection*.txt"
MAGEMin_output_file_list = glob.glob(MAGEMin_output_directory + "/" + MAGEMin_output_pattern)

# MAGEMin inputs
MAGEMin_mode = 0
MAGEMin_parallel = True
MAGEMin_parallel_nprocs = 6
MAGEMin_pressure_range = [30, 100, 1]
MAGEMin_temperature_range = [300, 1000, 10]
MAGEMin_composition = [41.49, 1.57, 4.824, 52.56, 5.88, 0.01, 0.25, 0.1, 0.1, 0.0, 0.0]

# Param to visualize
MAGEMin_param = "Rho[kg/m3]"

results = process_MAGEMin_files(MAGEMin_output_directory, MAGEMin_output_pattern)

for result in results:
    print(f"Point {result['Point']}:")
    print(f"Status: {result['Status']}")
    print(f"P: {result['P']}")
    print(f"T: {result['T']}")
    print(f"Gibbs: {result['Gibbs']}")
    print(f"BrNorm: {result['BrNorm']}")
    print(f"Gamma: {result['Gamma']}")
    print(f"Vp: {result['Vp']}")
    print(f"Vs: {result['Vs']}")
    print(f"Entropy: {result['Entropy']}")
    print(f"Stable Solutions: {result['StableSolutions']}")
    print(f"Stable Fractions: {result['StableFractions']}")
    print(f"Density: {result['Density']}")
    print(f"Compositional Var: {result['CompositionalVar']}")
    print(f"EM Fractions: {result['EMFractions']}")
    print(f"EM List: {result['EMList']}")
    print(f"Liquid Fraction: {result['LiquidFraction']}")
    print(f"Density of Full Assemblage: {result['DensityOfFullAssemblage']}")
    print(f"Density of Liquid: {result['DensityOfLiquid']}")
    print(f"Density of Solid: {result['DensityOfSolid']}")
    print(f"Density of Mixture: {result['DensityOfMixture']}")
    print()

# Run MAGEMin
#run_MAGEMin(
#    MAGEMin_program_path,
#    MAGEMin_input_path,
#    MAGEMin_output_directory,
#    MAGEMin_output_pattern,
#    MAGEMin_mode,
#    MAGEMin_parallel,
#    MAGEMin_parallel_nprocs,
#    MAGEMin_pressure_range,
#    MAGEMin_temperature_range,
#    MAGEMin_composition,
#    MAGEMin_param,
#    MAGEMin_plot=True,
#    MAGEMin_plot_fname="test.png",
#    verbose=True
#)

print("Done!")