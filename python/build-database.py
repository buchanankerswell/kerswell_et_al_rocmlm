from magemin import (
    parse_arguments_build_db,
    random_sample_for_MAGEMin,
    batch_sample_for_MAGEMin,
    normalize_sample,
    create_MAGEMin_input,
    run_MAGEMin
)

# Parse arguments
args = parse_arguments_build_db()

# Get argument values
P_range = args.Prange
T_range = args.Trange
source = args.source
strategy = args.strategy
n = args.n
k = args.k
parallel = args.parallel
nprocs = args.nprocs
seed = args.seed
out_dir = args.outdir

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Building with the following parameters:")
print(f"Prange: {P_range}")
print(f"Trange: {T_range}")
print(f"source: {source}")
print(f"strategy: {strategy}")
print(f"n: {n}")
print(f"k: {k}")
print(f"parallel: {parallel}")
print(f"nprocs: {nprocs}")
print(f"seed: {seed}")
print(f"out_dir: {out_dir}")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# Check arguments
if source not in ["earthchem"]:
    raise ValueError(
        "Invalid --source argument ...\n"
        "Use --source=earthchem"
    )
if strategy not in ["random", "batch"]:
    raise ValueError(
        "Sample strategy not recognized ...\n"
        "Use either --strategy \"random\" or --strategy \"batch\""
    )

# Sample database and run MAGEMin to compute stable assemblages
if strategy == "random":
    # Select sample randomly from Earthchem dataset and run MAGEMin n-times
    sample_ids, sample_comps = random_sample_for_MAGEMin(
        "assets/data/earthchem-ig.csv", n, seed
    )

    # Print info
    print("Sample IDs and compositions:")
    print(sample_ids)
    print(sample_comps)

    for sample_id, sample_comp in zip(sample_ids, sample_comps):
        # Normalize composition
        sample_norm = normalize_sample(sample_comp)

        # Write MAGEMin input
        run_name = sample_id
        create_MAGEMin_input(P_range,T_range, sample_norm, run_name=run_name)

        # Run MAGEMin
        run_MAGEMin(
            "MAGEMin/",
            run_name=run_name,
            comp_type="wt",
            parallel=parallel,
            nprocs=nprocs,
            out_dir=out_dir
        )

if strategy == "batch":
    # Split Earthchem database into n batches
    sample_ids, sample_comps = batch_sample_for_MAGEMin(
        "assets/data/earthchem-ig.csv", n, k
    )

    # Print info
    print("Sample IDs and compositions:")
    for sample_id, sample_comp in zip(sample_ids, sample_comps):
        print(sample_id)
        print(sample_comp)

        # Normalize composition
        sample_norm = normalize_sample(sample_comp)

        # Write MAGEMin input
        run_name = sample_id
        create_MAGEMin_input(
            P_range,
            T_range,
            sample_norm,
            run_name=run_name,
            out_dir=out_dir
        )

        # Run MAGEMin
        run_MAGEMin(
            "MAGEMin/",
            run_name=run_name,
            comp_type="wt",
            parallel=parallel,
            nprocs=nprocs,
            out_dir=out_dir
        )
