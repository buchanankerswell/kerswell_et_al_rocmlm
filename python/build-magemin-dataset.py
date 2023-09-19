from rocml import (
    parse_arguments,
    check_arguments,
    configure_magemin_model,
    run_magemin,
    get_comp_time
)

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "build-magemin-dataset.py")

# Load valid arguments
locals().update(valid_args)

# Configure MAGEMin model
configure_magemin_model(Pmin, Pmax, Tmin, Tmax, res, source, sampleid, normox, dataset)

# Run MAGEMin
run_magemin(sampleid, res, dataset)

# Get MAGEMin comp time and write to csv
get_comp_time(logfile, sampleid, dataset, res, nprocs)

print("build-magemin-dataset.py done!")