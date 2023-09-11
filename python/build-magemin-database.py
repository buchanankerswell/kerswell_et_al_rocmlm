from rocml import (
    run_MAGEMin,
    parse_arguments,
    check_arguments,
    normalize_sample,
    create_MAGEMin_input,
    get_benchmark_sample_for_MAGEMin
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-magemin-database.py")

# Load valid arguments
locals().update(valid_args)

# Get benchmarking composition
datafile = "assets/data/benchmark-samples.csv"
sample = get_benchmark_sample_for_MAGEMin(datafile, sampleid)

# Normalize composition
norm = normalize_sample(sample, normox)

# Transform units
Pmin, Pmax, Tmin, Tmax = Pmin * 10, Pmax * 10, Tmin - 273, Tmax - 273

# PT grid
Prange, Trange = [Pmin, Pmax, (Pmax-Pmin)/res], [Tmin, Tmax, (Tmax-Tmin)/res]

# Write MAGEMin input
create_MAGEMin_input(Prange, Trange, norm, 0, sampleid, outdir)

# Run MAGEMin
run_MAGEMin("MAGEMin/", sampleid, "wt", "ig", parallel, nprocs, outdir)