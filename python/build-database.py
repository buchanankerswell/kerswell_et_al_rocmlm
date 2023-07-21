import os
from magemin import (
    run_MAGEMin,
    parse_arguments,
    check_arguments,
    normalize_sample,
    create_MAGEMin_input,
    batch_sample_for_MAGEMin,
    random_sample_for_MAGEMin
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "run-benchmark.py")

# Load valid arguments
locals().update(valid_args)

# Transform units
Pmin, Pmax, Tmin, Tmax = Pmin * 10, Pmax * 10, Tmin - 273, Tmax - 273

# PT grid
Prange, Trange = [Pmin, Pmax, (Pmax-Pmin)/Pres], [Tmin, Tmax, (Tmax-Tmin)/Tres]

# MAGEMin oxide options
oxide_list_magemin = [
    "SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "Fe2O3", "Cr2O3", "H2O"
]

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
            sample_norm = normalize_sample(sample_comp, components=normox)

            # Print sample info
            print(sample_id)
            for component, value in zip(oxide_list_magemin, sample_norm):
                formatted_value = "{:.2f}".format(value)
                print(f"{component}: {formatted_value}")
            print("---------------------------------------------")

            # Write MAGEMin input
            create_MAGEMin_input(Prange, Trange, sample_norm, 0, sample_id, outdir)

            # Run MAGEMin
            run_MAGEMin("MAGEMin/", sample_id, "wt", "ig", parallel, nprocs, outdir)

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
            create_MAGEMin_input(Prange, Trange, sample_norm, 0, sample_id, outdir)

            # Run MAGEMin
            run_MAGEMin("MAGEMin/", sample_id, "wt", "ig", parallel, nprocs, outdir)

# Run single sample defined by user
if source == "sample":
    # Normalize composition
    sample_norm = normalize_sample(sample=sample_comp, components=normox)

    # Write MAGEMin input
    create_MAGEMin_input(Prange, Trange, sample_norm, 0, sample_id, outdir)

    # Run MAGEMin
    run_MAGEMin("MAGEMin/", sample_id, "wt", "ig", parallel, nprocs, outdir)
