import os
from magemin import (
    parse_arguments_build_db,
    check_non_matching_strings,
    random_sample_for_MAGEMin,
    batch_sample_for_MAGEMin,
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
sample_comp = args.comp
comp_type = args.frac
sample_id = args.sampleid
norm_ox = args.normox
source = args.source
strategy = args.strategy
n = args.n
k = args.k
parallel = args.parallel
nprocs = args.nprocs
seed = args.seed
out_dir = args.outdir

# MAGEMin oxide options
oxide_list_magemin = [
    "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "Fe2O3", "Cr2O3", "H2O"
]

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
if norm_ox != "all":
    if check_non_matching_strings(norm_ox, oxide_list_magemin):
        raise ValueError(
            "Invalid --normox argument ...\n"
            f"Can only normalize to oxides {oxide_list_magemin}"
            "Or --normox=all"
        )
print(f"Normalizing composition to: {norm_ox}")
if source == "earthchem":
    if strategy not in ["random", "batch"]:
        raise ValueError(
            "Invalid --strategy argument ...\n"
            "Use --source=earthchem or --source=sample"
        )
    print(f"strategy: {strategy}")
    print(f"n: {n}")
    print(f"k: {k}")
elif source == "sample":
    print(f"sample_id: {sample_id}")
    print("-------------")
    print("sample_comp:")
    for component, value in zip(oxide_list_magemin, sample_comp):
        formatted_value = "{:.2f}".format(value)
        print(f"{component}: {formatted_value}")
    print("-------------")
    if comp_type not in ["mol", "wt"]:
        raise ValueError(
            "Invalid --frac argument ...\n"
            "Use --frac=mol or --frac=wt"
        )
    print(f"comp_type: {comp_type}")
else:
    raise ValueError(
        "Invalid --source argument ...\n"
        "Use --source=earthchem"
    )
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
print(f"seed: {seed}")
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

# Sample database and run MAGEMin to compute stable assemblages
if source == "earthchem":
    # Random sampling from earthchem
    if strategy == "random":
        # Select sample randomly from Earthchem dataset and run MAGEMin n-times
        sample_ids, sample_comps = random_sample_for_MAGEMin(
            "assets/data/earthchem-samples.csv", n, seed
        )

        print("Sample IDs and compositions:")

        for sample_id, sample_comp in zip(sample_ids, sample_comps):
            # Normalize composition
            sample_norm = normalize_sample(sample_comp, components=norm_ox)

            # Print sample info
            print(sample_id)
            for component, value in zip(oxide_list_magemin, sample_norm):
                formatted_value = "{:.2f}".format(value)
                print(f"{component}: {formatted_value}")
            print("---------------------------------------------")

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

    # Batch sampling from earthchem
    if strategy == "batch":
        # Split Earthchem database into n batches
        sample_ids, sample_comps = batch_sample_for_MAGEMin(
            "assets/data/earthchem-samples.csv", n, k
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

# Run single sample defined by user
if source == "sample":
    # Normalize composition
    sample_norm = normalize_sample(sample=sample_comp, components=norm_ox)

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
        comp_type=comp_type,
        parallel=parallel,
        nprocs=nprocs,
        out_dir=out_dir
    )
