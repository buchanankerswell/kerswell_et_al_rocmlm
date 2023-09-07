import os
from magemin import (
    print_filepaths,
    visualize_earthchem_data,
    combine_plots_vertically,
    combine_plots_horizontally,
    visualize_training_PT_range,
    visualize_regression_metrics,
    visualize_benchmark_gfem_times
)

if os.path.exists("assets/data"):
    print("Plotting results to:")

    # Visualize Earthchem data
    visualize_earthchem_data("assets/data/earthchem-samples.csv")

    # Visualize Clapeyron slopes for 660 transition
    visualize_training_PT_range()

    # Visualize benchmark computation times
    visualize_benchmark_gfem_times("assets/data/benchmark-gfem-efficiency.csv")

    # Visualize regression metrics
    visualize_regression_metrics("assets/data/benchmark-mlms-metrics.csv")

    # First row
    combine_plots_horizontally(
        "figs/regression-inference-time-mean.png",
        "figs/regression-training-time-mean.png",
        "figs/temp1.png",
        caption1="a)",
        caption2="b)"
    )

    os.remove("figs/regression-inference-time-mean.png")
    os.remove("figs/regression-training-time-mean.png")

    # Second row
    combine_plots_horizontally(
        "figs/regression-rmse-test-mean.png",
        "figs/regression-rmse-valid-mean.png",
        "figs/temp2.png",
        caption1="c)",
        caption2="d)"
    )

    os.remove("figs/regression-rmse-test-mean.png")
    os.remove("figs/regression-rmse-valid-mean.png")

    # Stack rows
    combine_plots_vertically(
        "figs/temp1.png",
        "figs/temp2.png",
        "figs/benchmark-mlms-metrics.png",
        caption1="",
        caption2=""
    )

    os.remove("figs/temp1.png")
    os.remove("figs/temp2.png")

print_filepaths("figs")