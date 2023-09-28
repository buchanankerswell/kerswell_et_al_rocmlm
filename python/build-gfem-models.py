from rocml import parse_arguments, check_arguments, get_random_sampleids, build_gfem_models

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

print("build-gfem-models.py done!")