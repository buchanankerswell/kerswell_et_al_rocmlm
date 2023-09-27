import os
from rocml import (
    parse_arguments,
    check_arguments,
    visualize_training_dataset_design,
    visualize_benchmark_efficiency,
    visualize_rocml_performance,
    combine_plots_vertically,
    combine_plots_horizontally
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "visualize-other.py")

# Load valid arguments
locals().update(valid_args)

if os.path.exists("assets/data"):
    # Visualize Clapeyron slopes for 660 transition
    visualize_training_dataset_design(Pmin, Pmax, Tmin, Tmax)

    if os.path.exists(f"assets/data/benchmark-efficiency.csv"):
        # Visualize benchmark computation times
        visualize_benchmark_efficiency(figdir, "benchmark-efficiency.png")

    if os.path.exists(f"assets/data/rocml-performance.csv"):
        for t in targets:
            # Visualize rocml performance metrics
            visualize_rocml_performance(sampleid, t, res, figdir, "rocml")

            # First row
            combine_plots_horizontally(
                f"figs/rocml-inference-time-mean-{sampleid}-{res}.png",
                f"figs/rocml-training-time-mean-{sampleid}-{res}.png",
                "figs/temp1.png",
                caption1="a)",
                caption2="b)"
            )

            os.remove(f"figs/rocml-inference-time-mean-{sampleid}-{res}.png")
            os.remove(f"figs/rocml-training-time-mean-{sampleid}-{res}.png")

            # Second row
            combine_plots_horizontally(
                f"figs/rocml-rmse-test-mean-{t.replace('_', '-')}-{sampleid}-{res}.png",
                f"figs/rocml-rmse-val-mean-{t.replace('_', '-')}-{sampleid}-{res}.png",
                "figs/temp2.png",
                caption1="c)",
                caption2="d)"
            )

            os.remove(f"figs/rocml-rmse-test-mean-{t.replace('_', '-')}-{sampleid}-{res}.png")
            os.remove(f"figs/rocml-rmse-val-mean-{t.replace('_', '-')}-{sampleid}-{res}.png")

            # Stack rows
            combine_plots_vertically(
                "figs/temp1.png",
                "figs/temp2.png",
                f"figs/rocml-performance-{t.replace('_', '-')}.png",
                caption1="",
                caption2=""
            )

            os.remove("figs/temp1.png")
            os.remove("figs/temp2.png")

print("visualize-other.py done!")