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

# Get values
P_range = args.Prange
T_range = args.Trange
type = args.type
n = args.n
k = args.k
parallel = args.parallel
nprocs = args.nprocs
out_dir = args.out_dir

print("=============================================")
print("Using the following parameters:")
print(f"Prange: {args.Prange}")
print(f"Trange: {args.Trange}")
print(f"type: {args.type}")
print(f"n: {args.n}")
print(f"k: {args.k}")
print(f"nprocs: {args.nprocs}")
print(f"parallel: {args.parallel}")
print(f"out_dir: {args.out_dir}")
print("=============================================")

if type == "random":
    # Select sample randomly from Earthchem dataset and run MAGEMin n-times
    sample_ids, sample_comps = random_sample_for_MAGEMin("data/earthchem-ig.csv", n)

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
        run_MAGEMin("MAGEMin/", run_name=run_name, comp_type="wt")

elif type == "batch":
    # Split EC database into n batches
    sample_ids, sample_comps = batch_sample_for_MAGEMin("data/earthchem-ig.csv", n, k)

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

else:
    raise ValueError(
        "Type not recognized ...\n"
        "Use either --type \"random\" or --type \"batch\""
    )
