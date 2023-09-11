import os
from rocml import (
    print_filepaths,
    visualize_earthchem_data,
    combine_plots_vertically,
    combine_plots_horizontally,
    visualize_training_PT_range,
    visualize_rocml_performance,
    visualize_benchmark_gfem_times
)

if os.path.exists("assets/data"):
    # Visualize Earthchem data
    visualize_earthchem_data("assets/data/earthchem-samples.csv")

    # Visualize Clapeyron slopes for 660 transition
    visualize_training_PT_range()

    # Visualize benchmark computation times
    visualize_benchmark_gfem_times("assets/data/benchmark-gfem-efficiency.csv")

    # Visualize rocml performance metrics
    visualize_rocml_performance(
        "assets/data/benchmark-rocmls-performance.csv",
        sample_id="PUM",
        res=64
    )

    # First row
    combine_plots_horizontally(
        "figs/rocml-inference-time-mean.png",
        "figs/rocml-training-time-mean.png",
        "figs/temp1.png",
        caption1="a)",
        caption2="b)"
    )

    os.remove("figs/rocml-inference-time-mean.png")
    os.remove("figs/rocml-training-time-mean.png")

    # Second row
    combine_plots_horizontally(
        "figs/rocml-rmse-test-mean.png",
        "figs/rocml-rmse-valid-mean.png",
        "figs/temp2.png",
        caption1="c)",
        caption2="d)"
    )

    os.remove("figs/rocml-rmse-test-mean.png")
    os.remove("figs/rocml-rmse-valid-mean.png")

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

print("Figures:")
print_filepaths("figs")
print("visualize-other.py done!")