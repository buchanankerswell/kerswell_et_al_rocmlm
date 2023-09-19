from rocml import (
    parse_arguments,
    check_arguments,
    earthchem_samples_pca
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "earthchem-samples-pca.py")

# Load valid arguments
locals().update(valid_args)

earthchem_samples_pca(
    res=res,
    oxides=oxides,
    n_pca_components=npca,
    k_pca_clusters=kcluster,
    fig_dir=figdir,
    data_dir=datadir
)

print("earthchem-samples-pca.py done!")