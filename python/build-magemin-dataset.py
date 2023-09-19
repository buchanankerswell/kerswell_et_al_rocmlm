from rocml import (
    parse_arguments,
    check_arguments,
    configure_magemin,
    run_magemin,
    get_comp_time
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-magemin-dataset.py")

# Load valid arguments
locals().update(valid_args)

# Configure MAGEMin model
configure_magemin(Pmin, Pmax, Tmin, Tmax, res, source, sampleid, normox, dataset, outdir)

# Run MAGEMin
run_magemin(sampleid, res, dataset, emsonly, parallel, nprocs, outdir, verbose)

# Get MAGEMin comp time and write to csv
get_comp_time(logfile, sampleid, dataset, res, nprocs, datadir)

print("build-magemin-dataset.py done!")