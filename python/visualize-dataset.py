import os
from rocml import (
    parse_arguments,
    check_arguments,
    visualize_training_dataset,
    visualize_training_dataset_diff,
    compose_dataset_plots
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "visualize-dataset.py")

# Load valid arguments
locals().update(valid_args)

# Test for results
mgm_results_train = os.path.exists(f"runs/magemin_{sampleid}_train_{res}")
mgm_results_valid = os.path.exists(f"runs/magemin_{sampleid}_valid_{res}")
ppx_results_train = os.path.exists(f"runs/perplex_{sampleid}_train_{res}")
ppx_results_valid = os.path.exists(f"runs/perplex_{sampleid}_valid_{res}")

if mgm_results_train and mgm_results_valid:
    magemin = True

else:
    magemin = False

if ppx_results_train and ppx_results_valid:
    perplex = True

else:
    perplex = False

print(f"Plotting results for {sampleid} ...")

# Plot magemin output
if magemin:
    visualize_training_dataset("magemin", sampleid, res, "train", vistargets,
                               maskgeotherm, palette, figdir, verbose)

# Plot perplex output
if perplex:
    visualize_training_dataset("perplex", sampleid, res, "train", vistargets,
                               maskgeotherm, palette, figdir, verbose)

# Plot magemin perplex difference
if magemin and perplex:
    visualize_training_dataset_diff(sampleid, res, "train", vistargets,
                                    maskgeotherm, palette, figdir, verbose)

# Compose plots
compose_dataset_plots(magemin, perplex, sampleid, "train", res, vistargets, figdir, verbose)

print("visualize-dataset.py done!")