from rocml import parse_arguments, check_arguments, train_rocml, compose_rocml_plots

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-rocml-models.py")

# Load valid arguments
locals().update(valid_args)

# Train and visualize RocML models
for model in models:
    train_rocml("magemin", sampleid, res, targets, maskgeotherm, model, tune, epochs, batchp,
                kfolds, parallel, nprocs, seed, palette, f"{figdir}/{sampleid}_{res}", verbose)
    train_rocml("perplex", sampleid, res, targets, maskgeotherm, model, tune, epochs, batchp,
                kfolds, parallel, nprocs, seed, palette, f"{figdir}/{sampleid}_{res}", verbose)

# Compose plots
compose_rocml_plots(True, True, sampleid, res, models, targets, f"{figdir}/{sampleid}_{res}")

print("train-rocml-models.py done!")