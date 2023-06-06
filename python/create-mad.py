#!/usr/bin/env python

from functions import create_MAGEMin_input, run_MAGEMin

# Write MAGEMin input
create_MAGEMin_input(
    P_range=[10, 100, 10],
    T_range=[100, 1000, 100],
    composition=[41.49, 1.57, 4.824, 52.56, 5.88, 0.01, 0.25, 0.1, 0.1, 0.0, 0.0],
    run_name="test"
)

# Run MAGEMin
run_MAGEMin(program_path="MAGEMin/", run_name="test", comp_type="mol", verbose=True)