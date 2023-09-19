import os
from rocml import (
    parse_arguments,
    check_arguments,
    combine_plots_vertically,
    combine_plots_horizontally,
    visualize_training_PT_range,
    visualize_rocml_performance,
    visualize_benchmark_efficiency
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "visualize-other.py")

# Load valid arguments
locals().update(valid_args)

if os.path.exists("assets/data"):
    # Visualize Clapeyron slopes for 660 transition
    visualize_training_PT_range()

    # Visualize benchmark computation times
    visualize_benchmark_efficiency("assets/data/benchmark-efficiency.csv")

    for param in params:
        # Visualize rocml performance metrics
        visualize_rocml_performance(sample_id=sampleid, parameter=param, res=res)

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
            f"figs/rocml-rmse-test-mean-{param}-{sampleid}-{res}.png",
            f"figs/rocml-rmse-val-mean-{param}-{sampleid}-{res}.png",
            "figs/temp2.png",
            caption1="c)",
            caption2="d)"
        )

        os.remove(f"figs/rocml-rmse-test-mean-{param}-{sampleid}-{res}.png")
        os.remove(f"figs/rocml-rmse-val-mean-{param}-{sampleid}-{res}.png")

        # Stack rows
        combine_plots_vertically(
            "figs/temp1.png",
            "figs/temp2.png",
            f"figs/benchmark-rocmls-metrics-{param}.png",
            caption1="",
            caption2=""
        )

        os.remove("figs/temp1.png")
        os.remove("figs/temp2.png")

print("visualize-other.py done!")