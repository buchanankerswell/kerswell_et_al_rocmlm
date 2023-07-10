import os
from magemin import visualize_training_PT_range, visualize_benchmark_comp_times

# Visualize Clapeyron slopes for 660 transition
visualize_training_PT_range()

# Visualize benchmark computation times
if os.path.exists("assets/data"):
    comp_times = "assets/data/benchmark-times.csv"
    # Plot benchmark comp times
    visualize_benchmark_comp_times(comp_times)