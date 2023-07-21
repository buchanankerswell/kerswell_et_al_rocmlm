from magemin import (
    run_MAGEMin,
    parse_arguments,
    check_arguments,
    normalize_sample,
    create_MAGEMin_input,
    get_benchmark_sample_for_MAGEMin
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "benchmark.py")

# Load valid arguments
locals().update(valid_args)

# Get benchmarking composition
datafile = "assets/data/benchmark-comps.csv"
sample = get_benchmark_sample_for_MAGEMin(datafile, sampleid)

# Normalize composition
norm = normalize_sample(sample, normox)

# Transform units
Pmin, Pmax, Tmin, Tmax = Pmin * 10, Pmax * 10, Tmin - 273, Tmax - 273

# PT grid
Prange, Trange = [Pmin, Pmax, (Pmax-Pmin)/Pres], [Tmin, Tmax, (Tmax-Tmin)/Tres]

# Write MAGEMin input
create_MAGEMin_input(Prange, Trange, norm, 0, sampleid, outdir)

# Run MAGEMin
run_MAGEMin("MAGEMin/", sampleid, "wt", "ig", parallel, nprocs, outdir)