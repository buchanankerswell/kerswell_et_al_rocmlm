from rocml import (
    parse_arguments,
    check_arguments,
    configure_perplex_model,
    run_perplex,
    get_comp_time
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-perplex-dataset.py")

# Load valid arguments
locals().update(valid_args)

# Configure Perple_X model
configure_perplex_model(Pmin, Pmax, Tmin, Tmax, res, source, sampleid,
                        normox, dataset, emsonly, configdir, perplexdir, outdir)

# Run Perple_X
run_perplex(sampleid, dataset, res, emsonly, perplexdir, outdir, verbose)

# Get Perple_X comp time and write to csv
get_comp_time(logfile, sampleid, dataset, res, nprocs, datadir)

print("build-perplex-dataset.py done!")