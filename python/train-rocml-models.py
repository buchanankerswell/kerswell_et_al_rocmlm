from scripting import parse_arguments, check_arguments
from gfem import get_random_sampleids, GFEMModel
from rocml import RocML
from visualize import visualize_ml_model, compose_ml_plots

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "train-rocml-models.py")

# Load valid arguments
locals().update(valid_args)

# Fetch GFEM models
models = GFEMModel(["perplex"], ["train", "valid"], source, nsamples, res)

# Parse GFEM models
mage_train = [m for m in models if m.program == "magemin" and m.dataset == "train"]
mage_valid = [m for m in models if m.program == "magemin" and m.dataset == "valid"]
perp_train = [m for m in models if m.program == "perplex" and m.dataset == "train"]
perp_valid = [m for m in models if m.program == "perplex" and m.dataset == "valid"]

for mt, mv, pt, pv in zip(mage_train, mage_valid, perp_train, perp_valid):
    for m in mlmodels:
        # Init RocML models
        mage_rocml = RocML(mt, mv, m, tune, epochs, batchprop, kfolds, parallel, nprocs,
                           seed, palette, verbose)
        perp_rocml = RocML(pt, pv, m, tune, epochs, batchprop, kfolds, parallel, nprocs,
                           seed, palette, verbose)

        # Train RocML models
        mage_rocml.train_ml_model()
        perp_rocml.train_ml_model()

        # Visualize RocML models
        visualize_ml_model(mage_rocml)
        visualize_ml_model(perp_rocml)

for s in sampleids:
    # Compose plots
    compose_ml_plots(True, True, s, res, mlmodels, targets, f"{figdir}/{s}_{res}")

print("train-rocml-models.py done!")