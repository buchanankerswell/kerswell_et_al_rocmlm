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

# Run magemin and perplex
for program in ["magemin", "perplex"]:
    build_gfem_models(program, Pmin, Pmax, Tmin, Tmax, res, source, sampleids, normox,
                      parallel, nprocs, verbose)

# Visualize results
for sampleid in sampleids:
    # Plot magemin output
    visualize_training_dataset("magemin", sampleid, res, "train", targets, maskgeotherm,
                               palette, f"{figdir}/{sampleid}_{res}", verbose)

    # Plot perplex output
    visualize_training_dataset("perplex", sampleid, res, "train", targets, maskgeotherm,
                               palette, f"{figdir}/{sampleid}_{res}", verbose)

    # Plot magemin perplex difference
    visualize_training_dataset_diff(sampleid, res, "train", targets, maskgeotherm, palette,
                                    f"{figdir}/{sampleid}_{res}", verbose)

    # Compose plots
    compose_dataset_plots(True, True, sampleid, "train", res, targets,
                          f"{figdir}/{sampleid}_{res}", verbose)

# Visualize Clapeyron slopes for 660 transition
visualize_training_dataset_design(Pmin, Pmax, Tmin, Tmax, f"{figdir}/other")

# Visualize benchmark computation times
visualize_benchmark_efficiency(f"{figdir}/other", "benchmark-efficiency.png")

print("build-gfem-models.py done!")