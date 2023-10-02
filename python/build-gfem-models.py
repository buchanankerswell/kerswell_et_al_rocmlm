from rocml import (parse_arguments,
                   check_arguments,
                   get_random_sampleids,
                   build_gfem_models,
                   visualize_training_dataset,
                   visualize_training_dataset_diff,
                   compose_dataset_plots,
                   visualize_training_dataset_design,
                   visualize_benchmark_efficiency)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-gfem-models.py")

# Load valid arguments
locals().update(valid_args)

# Samples
if benchmarks:
    source = "assets/data/benchmark-samples.csv"
    sampleids = ["PUM", "DMM", "NMORB", "RE46"]

elif not benchmarks:
    source = f"assets/data/synthetic-samples-pca{npca}-clusters13.csv"
    sampleids = get_random_sampleids(source, n=nsamples)

# Initiate empty list to store models
models = []

# Build models
for program in ["magemin", "perplex"]:
    models.extend(build_gfem_models(program, Pmin, Pmax, Tmin, Tmax, res, source, sampleids,
                                    normox, targets, maskgeotherm, parallel, nprocs, verbose))

# Visualize results
for model in models:
    visualize_training_dataset(model, palette)

# Visualize diffs
magemin_models = [model for model in models if model.program == "magemin"]
perplex_models = [model for model in models if model.program == "perplex"]

for magemin_model, perplex_model in zip(magemin_models, perplex_models):
    visualize_training_dataset_diff(magemin_model, perplex_model, palette)

# Compose plots
for sampleid in sampleids:
    for dataset in ["train", "valid"]:
        compose_dataset_plots(True, True, sampleid, dataset, res, targets,
                              f"{figdir}/{sampleid}_{res}", verbose)

# Visualize Clapeyron slopes for 660 transition
visualize_training_dataset_design(Pmin, Pmax, Tmin, Tmax, f"{figdir}/other")

# Visualize benchmark computation times
visualize_benchmark_efficiency(f"{figdir}/other", "gfem-efficiency.png")

print("build-gfem-models.py done!")