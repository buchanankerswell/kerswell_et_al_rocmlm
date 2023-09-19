from rocml import (
    parse_arguments,
    check_arguments,
    pca_mixing_arrays
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "earthchem-pca-mixing-arrays.py")

# Load valid arguments
locals().update(valid_args)

pca_mixing_arrays(
    res=res,
    oxides=oxides,
    n_pca_components=npca,
    k_pca_clusters=kcluster,
    fig_dir=figdir,
    data_dir=datadir
)

print("earthchem-pca-mixing-arrays.py done!")