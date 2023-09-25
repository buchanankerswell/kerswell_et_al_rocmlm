from rocml import (
    parse_arguments,
    check_arguments,
    pca_mixing_arrays
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "make-pca-arrays.py")

# Load valid arguments
locals().update(valid_args)

# Make PCA mixing arrays
pca_mixing_arrays(res, oxides, npca, kcluster, fig_dir=figdir)

print("make-pca-arrays.py done!")