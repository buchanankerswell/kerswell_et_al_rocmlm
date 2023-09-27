import os
from rocml import parse_arguments, check_arguments, train_rocml, compose_rocml_plots

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-benchmark-rocmls.py")

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

# Train and visualize RocML models
for model in models:
    if magemin:
        train_rocml("magemin", sampleid, res, targets, maskgeotherm, model, tune,
                    kfolds, parallel, nprocs, seed, palette, figdir, verbose)
    if perplex:
        train_rocml("perplex", sampleid, res, targets, maskgeotherm, model, tune,
                    kfolds, parallel, nprocs, seed, palette, figdir, verbose)

# Compose plots
compose_rocml_plots(magemin, perplex, sampleid, models, targets, figdir)

print("train-benchmark-rocmls.py done!")