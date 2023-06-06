#!/usr/bin/env python

from functions import process_MAGEMin_files, plot_pseudosection

# Process MAGEMin output files
results = process_MAGEMin_files(run_name="test")

# Plot
parameter = "DensityOfFullAssemblage"
plot_pseudosection(results, parameter, filename="test.png")