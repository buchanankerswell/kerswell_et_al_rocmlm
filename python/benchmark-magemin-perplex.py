import os
from magemin import (
    parse_arguments_build_db,
    check_non_matching_strings,
    get_benchmark_sample_for_MAGEMin,
    normalize_sample,
    create_MAGEMin_input,
    run_MAGEMin
)

# Parse arguments
args = parse_arguments_build_db()

# Get argument values
P_min = args.Pmin
P_max = args.Pmax
P_res = args.Pres
T_min = args.Tmin
T_max = args.Tmax
T_res = args.Tres
sample_id = args.sampleid
norm_ox = args.normox
parallel = args.parallel
nprocs = args.nprocs
out_dir = args.outdir

# MAGEMin oxide options
oxide_list_magemin = [
    "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "Fe2O3", "Cr2O3", "H2O"
]

# Read benchmarking samples
datafile = "assets/data/benchmark-samples.csv"
benchmark_sample_ids = ["PUM", "DMM", "RE46", "NMORB1", "NMORB2"]

# Get benchmark sample composition
sample_comp = get_benchmark_sample_for_MAGEMin(datafile, sample_id)

# Normalize composition
sample_norm = normalize_sample(sample=sample_comp, components=norm_ox)

# Check arguments and print
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Building MAGEMin database with:")
if P_res > 128 or T_res > 128:
    raise ValueError(
        "Invalid --Pres or --Tres argument ...\n"
        "--Pres and --Tres must be <= 128"
    )
print(f"Prange: [{P_min}, {P_max}, {P_res}]")
print(f"Trange: [{T_min}, {T_max}, {T_res}]")
if sample_id not in benchmark_sample_ids:
    raise ValueError(
        f"Sample ID {sample_id} not {datafile} ...\n"
        f"Options: {benchmark_sample_ids}"
    )
print(f"sample_id: {sample_id}")
print("-------------")
print("sample_comp (normalized):")
for component, value in zip(oxide_list_magemin, sample_norm):
    formatted_value = "{:.2f}".format(value)
    print(f"{component}: {formatted_value}")
print("-------------")
if norm_ox != "all":
    if check_non_matching_strings(norm_ox, oxide_list_magemin):
        raise ValueError(
            "Invalid --normox argument ...\n"
            f"Can only normalize to oxides {oxide_list_magemin}"
        )
print(f"Normalizing composition to: {norm_ox}")
if not isinstance(parallel, bool):
    raise ValueError(
        "Invalid --parallel argument ...\n"
        "--parallel must be either True or False"
    )
print(f"parallel: {parallel}")
if nprocs > os.cpu_count():
    raise ValueError(
        "Invalid --nprocs argument ...\n"
        f"--nprocs cannot be greater than cores on system ({os.cpu_count}) ..."
    )
print(f"nprocs: {nprocs}")
if len(out_dir) > 55:
    raise ValueError(
        "Invalid --outdir argument ...\n"
        f"--outdir cannot be greater than 55 characters ..."
        f"{out_dir}"
    )
print(f"out_dir: {out_dir}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# PT grid
P_range = [P_min, P_max, (P_max-P_min)/P_res]
T_range = [T_min, T_max, (T_max-T_min)/T_res]

# Write MAGEMin input
create_MAGEMin_input(
    P_range,
    T_range,
    sample_norm,
    run_name=sample_id,
    out_dir=out_dir
)

# Run MAGEMin
run_MAGEMin(
    "MAGEMin/",
    run_name=sample_id,
    comp_type="wt",
    parallel=parallel,
    nprocs=nprocs,
    out_dir=out_dir
)