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
models = GFEMModel(source=source, nsamples=res, res=res)

# Parse GFEM models
mage_train = [m for m in models if m.program == "magemin" and m.dataset == "train"]
mage_valid = [m for m in models if m.program == "magemin" and m.dataset == "valid"]
perp_train = [m for m in models if m.program == "perplex" and m.dataset == "train"]
perp_valid = [m for m in models if m.program == "perplex" and m.dataset == "valid"]

# Initialize lists
mage_rocmls, perp_rocmls = [], []

for mt, mv, pt, pv in zip(mage_train, mage_valid, perp_train, perp_valid):
    for ml_model in ["KN", "RF", "DT", "NN1", "NN2", "NN3"]:
        # Init RocML models
        mage_rocml, perp_rocml = RocML(mt, mv, ml_model), RocML(pt, pv, ml_model)

        # Train RocML models
        mage_rocml.train_ml_model()
        perp_rocml.train_ml_model()

        # Visualize RocML models
        visualize_ml_model(mage_rocml)
        visualize_ml_model(perp_rocml)

        # Append to list
        mage_rocmls.append(mage_rocml)
        perp_rocmls.append(perp_rocml)

# Compose plots
compose_ml_plots(mage_rocmls, perp_rocmls)

print("train-rocml-models.py done!")