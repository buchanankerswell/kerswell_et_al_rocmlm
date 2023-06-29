import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset from the CSV file
data = pd.read_csv("~/Working/kerswell_et_al_madnn/assets/data/benchmark-comp-times.csv")

# Set plot style and settings
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "0.5"

# Create a dictionary to map methods to marker styles and line styles
method_styles = {
    "magemin": {"marker": "o", "linestyle": "-"},
    "perplex": {"marker": "s", "linestyle": "--"}
}

# Create a dictionary to map samples to colors
sample_colors = {
    "DMM": "blue",
    "NMORB1": "green",
    "NMORB2": "red",
    "PUM": "orange",
    "RE46": "purple"
}

# Get max resolution
data["maxres"] = data[["tres", "pres"]].max(axis=1)

# Group the data by method and sample
grouped_data = data.groupby(["method", "sample"])

# Plot the data
for group, group_data in grouped_data:
    method_val, sample_val = group
    marker_style = method_styles[method_val]["marker"]
    linestyle_val = method_styles[method_val]["linestyle"]
    color_val = sample_colors[sample_val]

    # Filter out rows with missing time values
    filtered_data = group_data.dropna(subset=["time"])

    # Extract x and y values
    x = filtered_data["maxres"]
    y = filtered_data["time"]

    # Plot the data points and connect them with lines
    plt.plot(
        x, y,
        marker=marker_style,
        color=color_val,
        linestyle=linestyle_val,
        label=f"{method_val} {sample_val}"
    )

# Set labels and title
plt.xlabel("PT Grid Resolution")
plt.ylabel("Time (s)")
plt.title("Gibbs Minimization Efficiency")

# Set x-axis ticks
plt.xticks([8, 16, 32, 64, 128])

# Set legend
plt.legend()

# Use tight layout
plt.tight_layout()

# Save the plot to a file with 330 DPI
plt.savefig("figs/benchmark-comp-times.png", dpi=330)
