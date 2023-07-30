import os
from magemin import (
    print_filepaths,
    visualize_earthchem_data,
    visualize_training_PT_range,
    visualize_benchmark_comp_times
)

if os.path.exists("assets/data"):
    print("Plotting results to:")

    # Visualize Earthchem data
    visualize_earthchem_data("assets/data/earthchem-samples.csv")

    # Visualize Clapeyron slopes for 660 transition
    visualize_training_PT_range()

    # Visualize benchmark computation times
    visualize_benchmark_comp_times("assets/data/benchmark-times.csv")

    # Visualize regression metrics
    visualize_regression_metrics("assets/data/regression-info.csv")

print_filepaths("figs")