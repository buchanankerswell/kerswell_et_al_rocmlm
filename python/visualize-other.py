import os
from magemin import (
    print_filepaths,
    visualize_earthchem_data,
    visualize_training_PT_range,
    visualize_benchmark_comp_times
)

print("Plotting results to:")

# Visualize Earthchem data
visualize_earthchem_data("assets/data/earthchem-samples.csv")

# Visualize Clapeyron slopes for 660 transition
visualize_training_PT_range()

# Visualize benchmark computation times
if os.path.exists("assets/data"):
    comp_times = "assets/data/benchmark-times.csv"
    # Plot benchmark comp times
    visualize_benchmark_comp_times(comp_times)

print_filepaths("figs")