#######################################################
## .0.              Load Libraries               !!! ##
#######################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utilities !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import time
import shutil
import joblib
import traceback
import itertools
from tqdm import tqdm

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# parallel computing !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing as mp

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataframes and arrays !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plotting !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# machine learning !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score

#######################################################
## .1.             Helper Functions              !!! ##
#######################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# get unique value !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_unique_value(input_list):
    """
    """
    unique_value = input_list[0]

    for item in input_list[1:]:
        if item != unique_value:
            raise ValueError("Not all values are the same!")

    return unique_value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# train rocmlms !!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_rocmlms(gfem_models, ml_models=["DT", "KN", "RF", "NN1", "NN2", "NN3"],
                  training_features=["F_MELT_FRAC"], training_targets=["rho", "Vp", "Vs"],
                  tune=True, epochs=100, batchprop=0.2, kfolds=os.cpu_count(), parallel=True,
                  nprocs=os.cpu_count(), seed=42, palette="bone", verbose=1):
    """
    """
    # Check for models
    if not gfem_models:
        raise Exception("No GFEM models to compile!")

    # Get model metadata
    program = get_unique_value([m.program for m in gfem_models if m.dataset == "train"])
    sample_ids = [m.sample_id for m in gfem_models if m.dataset == "train"]
    res = get_unique_value([m.res for m in gfem_models if m.dataset == "train"])
    targets = get_unique_value([m.targets for m in gfem_models if m.dataset == "train"])
    mask_geotherm = get_unique_value([m.mask_geotherm for m in gfem_models if
                                      m.dataset == "train"])
    features = get_unique_value([m.features for m in gfem_models if m.dataset == "train"])
    oxides_exclude = get_unique_value([m.oxides_exclude for m in gfem_models if
                                       m.dataset == "train"])
    oxides = get_unique_value([m.oxides_system for m in gfem_models if m.dataset == "train"])
    subset_oxides = [oxide for oxide in oxides if oxide not in oxides_exclude]
    feature_list = subset_oxides + features

    # Get all PT arrays
    pt_train = np.stack([m.feature_array for m in gfem_models if m.dataset == "train"])
    pt_train_unmasked = np.stack([m.feature_array_unmasked for m in gfem_models if
                                  m.dataset == "train"])
    pt_valid = np.stack([m.feature_array for m in gfem_models if m.dataset == "valid"])
    pt_valid_unmasked = np.stack([m.feature_array_unmasked for m in gfem_models if
                                  m.dataset == "valid"])

    # Select features
    feature_indices = [i for i, feature in enumerate(feature_list) if feature in
                       training_features]

    # Get sample features
    feat_train, feat_valid = [], []

    for m in gfem_models:
        if m.dataset == "train":
            selected_features = [m.sample_features[i] for i in feature_indices]
            feat_train.append(selected_features)

        elif m.dataset == "valid":
            selected_features = [m.sample_features[i] for i in feature_indices]
            feat_valid.append(selected_features)

    feat_train = np.array(feat_train)
    feat_valid = np.array(feat_valid)

    # Tile features to match PT array shape
    feat_train = np.tile(feat_train[:, np.newaxis, :], (1, pt_train.shape[1], 1))
    feat_valid = np.tile(feat_valid[:, np.newaxis, :], (1, pt_valid.shape[1], 1))

    # Combine features
    combined_train = np.concatenate((feat_train, pt_train), axis=2)
    combined_train_unmasked = np.concatenate((feat_train, pt_train_unmasked), axis=2)
    combined_valid = np.concatenate((feat_valid, pt_valid), axis=2)
    combined_valid_unmasked = np.concatenate((feat_valid, pt_valid_unmasked), axis=2)

    # Flatten features
    feature_train = combined_train.reshape(-1, combined_train.shape[-1])
    feature_train_unmasked = combined_train_unmasked.reshape(-1, combined_train.shape[-1])
    feature_valid = combined_valid.reshape(-1, combined_valid.shape[-1])
    feature_valid_unmasked = combined_valid_unmasked.reshape(-1, combined_valid.shape[-1])

    # Define target indices
    target_indices = [targets.index(target) for target in training_targets]
    targets = [target for target in training_targets]

    # Get target arrays
    target_train = np.stack([m.target_array for m in gfem_models if m.dataset == "train"])
    target_train_unmasked = np.stack([m.target_array_unmasked for m in gfem_models if
                                      m.dataset == "train"])
    target_valid = np.stack([m.target_array for m in gfem_models if m.dataset == "valid"])
    target_valid_unmasked = np.stack([m.target_array_unmasked for m in gfem_models if
                                      m.dataset == "valid"])

    # Flatten targets
    target_train = target_train.reshape(-1, target_train.shape[-1])
    target_train_unmasked = target_train_unmasked.reshape(-1, target_train.shape[-1])
    target_valid = target_valid.reshape(-1, target_train.shape[-1])
    target_valid_unmasked = target_valid_unmasked.reshape(-1, target_train.shape[-1])

    # Select training targets
    target_train = target_train[:, target_indices]
    target_train_unmasked = target_train_unmasked[:, target_indices]
    target_valid = target_valid[:, target_indices]
    target_valid_unmasked = target_valid_unmasked[:, target_indices]

    # Get geotherm mask
    geotherm_mask_train = np.stack([m._create_geotherm_mask() for m in gfem_models if
                                    m.dataset == "train"]).flatten()
    geotherm_mask_valid = np.stack([m._create_geotherm_mask() for m in gfem_models if
                                    m.dataset == "valid"]).flatten()

    # Define array shapes
    M = int(len(gfem_models) / 2)
    W = int((res+1)**2)
    w = int(np.sqrt(W))
    F = int(len(training_features) + 2)
    T = int(len(targets))
    shape_feature = (M, W, F)
    shape_feature_square = (M, w, w, F)
    shape_target = (M, W, T)
    shape_target_square = (M, w, w, T)

    # Train rocmlm models
    rocmlms = []

    for model in ml_models:
        rocmlm = RocMLM(program, sample_ids, res, targets, mask_geotherm, feature_train,
                        feature_train_unmasked, target_train, target_train_unmasked,
                        feature_valid, feature_valid_unmasked, target_valid,
                        target_valid_unmasked, shape_feature, shape_feature_square,
                        shape_target, shape_target_square, geotherm_mask_train,
                        geotherm_mask_valid, model, tune, epochs, batchprop, kfolds,
                        parallel, nprocs, seed, palette, verbose)

        # Check for pretrained model
        rocmlm._check_pretrained_model()

        if rocmlm.ml_model_trained:
            pretrained_rocmlm = joblib.load(rocmlm.rocmlm_path)
            rocmlms.append(pretrained_rocmlm)

        else:
            # Train rocmlm
            rocmlm.train()
            rocmlms.append(rocmlm)

            # Save rocmlm
            with open(rocmlm.rocmlm_path, "wb") as file:
                joblib.dump(rocmlm, file)

            # Save ml model only
            with open(rocmlm.ml_model_only_path, "wb") as file:
                joblib.dump(rocmlm.ml_model_only, file)

    return rocmlms

#######################################################
## .2.               RocMLM Class                 !!! ##
#######################################################
class RocMLM:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # init !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, program, sample_ids, res, targets, mask_geotherm, feature_train,
                 feature_train_unmasked, target_train, target_train_unmasked, feature_valid,
                 feature_valid_unmasked, target_valid, target_valid_unmasked, shape_feature,
                 shape_feature_square, shape_target, shape_target_square,
                 geotherm_mask_train, geotherm_mask_valid,  ml_model, tune=True, epochs=100,
                 batchprop=0.2, kfolds=os.cpu_count(), parallel=True,
                 nprocs=os.cpu_count(), seed=42, palette="bone", verbose=1):
        """
        """
        # Input
        self.program = program
        self.sample_ids = sample_ids
        self.res = res
        self.targets = targets
        self.mask_geotherm = mask_geotherm
        self.shape_feature = shape_feature
        self.shape_feature_square = shape_feature_square
        self.shape_target = shape_target
        self.shape_target_square = shape_target_square
        self.geotherm_mask_train = geotherm_mask_train
        self.geotherm_mask_valid = geotherm_mask_valid
        if ml_model == "KN":
            ml_model_label = "K Nearest"
        elif ml_model == "RF":
            ml_model_label = "Random Forest"
        elif ml_model == "DT":
            ml_model_label = "Decision Tree"
        elif ml_model == "NN1":
            ml_model_label = "Neural Net 1L"
        elif ml_model == "NN2":
            ml_model_label = "Neural Net 2L"
        elif ml_model == "NN3":
            ml_model_label = "Neural Net 3L"
        self.ml_model_label = ml_model
        self.ml_model_label_full = ml_model_label
        self.tune = tune
        self.epochs = epochs
        self.batchprop = batchprop
        self.kfolds = kfolds
        self.parallel = parallel
        if not parallel:
            self.nprocs = 1
        elif parallel:
            if nprocs is None or nprocs > os.cpu_count():
                self.nprocs = os.cpu_count()
            else:
                self.nprocs = nprocs
        self.seed = seed
        self.palette = palette
        self.verbose = verbose
        self.model_out_dir = f"rocmlms"
        if any(sample in sample_ids for sample in ["PUM", "DMM"]):
            self.model_prefix = f"{self.program[:4]}-benchmark-{self.ml_model_label}"
            self.fig_dir = f"figs/rocmlm/{self.program[:4]}_benchmark_{self.ml_model_label}"
        else:
            self.model_prefix = (f"{self.program[:4]}-{self.sample_ids[0][:2]}-"
                                 f"{self.ml_model_label}")
            self.fig_dir = (f"figs/rocmlm/{self.program[:4]}_{self.sample_ids[0][:2]}_"
                            f"{self.ml_model_label}")
        self.rocmlm_path = f"{self.model_out_dir}/{self.model_prefix}.pkl"
        self.ml_model_only_path = f"{self.model_out_dir}/{self.model_prefix}-model-only.pkl"

        # Check for figs directory
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir, exist_ok=True)

        # Feature and target arrays
        self.feature_train = feature_train
        self.feature_train_unmasked = feature_train_unmasked
        self.target_train = target_train
        self.target_train_unmasked = target_train_unmasked
        self.feature_valid = feature_valid
        self.feature_valid_unmasked = feature_valid_unmasked
        self.target_valid = target_valid
        self.target_valid_unmasked = target_valid_unmasked

        # ML model definition and tuning
        self.ml_model = None
        self.ml_model_tuned = False

        # Cross validation performance metrics
        self.cv_info = {}
        self.ml_model_cross_validated = False

        # Square arrays for visualizations
        self.feature_square = np.array([])
        self.target_square = np.array([])
        self.prediction_square = np.array([])
        self.ml_model_trained = False
        self.ml_model_only = None
        self.ml_model_training_error = False
        self.ml_model_error = None

        # Set np array printing option
        np.set_printoptions(precision=3, suppress=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # check pretrained model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_pretrained_model(self):
        """
        """
        # Check for existing model build
        if os.path.exists(self.model_out_dir):
            if os.path.exists(self.rocmlm_path):
                self.ml_model_trained = True

                if self.verbose >= 1:
                    print(f"Found pretrained model {self.rocmlm_path}!")

        else:
            os.makedirs(self.model_out_dir, exist_ok=True)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # scale arrays !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _scale_arrays(self, feature_array, target_array):
        """
        """
        # Reshape the features and targets arrays
        X = feature_array
        y = target_array

        # Replace inf with nan
        X[~np.isfinite(X)] = np.nan
        y[~np.isfinite(y)] = np.nan

        # Create nan mask
        mask = np.any(np.isnan(y), axis=1)

        # Remove nans
        X, y = X[~mask,:], y[~mask,:]

        # Check for infinity in input data
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Input data contains NaN or infinity values.")

        # Initialize scalers
        scaler_X, scaler_y = StandardScaler(), StandardScaler()

        # Scale features array
        X_scaled = scaler_X.fit_transform(X)

        # Scale the target array
        y_scaled = scaler_y.fit_transform(y)

        return X, y, scaler_X, scaler_y, X_scaled, y_scaled

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # configure ml model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _configure_ml_model(self):
        """
        """
        # Get self attributes
        model_label = self.ml_model_label
        model_prefix = self.model_prefix
        feature_train = self.feature_train
        target_train = self.target_train
        tune = self.tune
        seed = self.seed
        epochs = self.epochs
        batchprop = self.batchprop
        nprocs = self.nprocs
        verbose = self.verbose

        # Check for training features
        if feature_train.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_train.size == 0:
            raise Exception("No training targets!")

        # Scale training dataset
        X_train, y_train, scaler_X_train, scaler_y_train, X_scaled_train, y_scaled_train = \
            self._scale_arrays(feature_train, target_train)

        # Define NN layer sizes
        nn_L1 = int(max(y_scaled_train.shape[0] * 0.01, 8))
        nn_L2 = int(max(y_scaled_train.shape[0] * 0.02, 16))
        nn_L3 = int(max(y_scaled_train.shape[0] * 0.05, 32))

        print(f"Configuring model {model_prefix} ...")

        if tune:
            # Define ML model and grid search param space for hyperparameter tuning
            print("Tuning RocMLM model ...")

            if model_label == "KN":
                model = KNeighborsRegressor()

                param_grid = dict(n_neighbors=[2, 4, 8], weights=["uniform", "distance"])

            elif model_label == "RF":
                model = RandomForestRegressor(random_state=seed)

                param_grid = dict(n_estimators=[400, 800, 1200],
                                  max_features=[1, 2, 3],
                                  min_samples_leaf=[1, 2, 3],
                                  min_samples_split=[2, 4, 6])

            elif model_label == "DT":
                model = DecisionTreeRegressor(random_state=seed)

                param_grid = dict(splitter=["best", "random"],
                                  max_features=[1, 2, 3],
                                  min_samples_leaf=[1, 2, 3],
                                  min_samples_split=[2, 4, 6])

            elif model_label == "NN1":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     batch_size=max(int(len(y_scaled_train) * batchprop), 8))

                param_grid = dict(hidden_layer_sizes=[(nn_L1), (nn_L2), (nn_L3)],
                                  learning_rate_init=[0.0001, 0.0005, 0.001])

            elif model_label == "NN2":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     batch_size=max(int(len(y_scaled_train) * batchprop), 8))

                param_grid = dict(hidden_layer_sizes=[(nn_L1, nn_L2),
                                                      (nn_L2, nn_L2),
                                                      (nn_L3, nn_L2)],
                                  learning_rate_init=[0.0001, 0.0005, 0.001])

            elif model_label == "NN3":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     batch_size=max(int(len(y_scaled_train) * batchprop), 8))

                param_grid = dict(hidden_layer_sizes=[(nn_L1, nn_L2, nn_L1),
                                                      (nn_L2, nn_L2, nn_L1),
                                                      (nn_L3, nn_L2, nn_L1)],
                                  learning_rate_init=[0.0001, 0.0005, 0.001])

            # K-fold cross validation
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)

            # Perform grid search hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid=param_grid, cv=kf,
                                       scoring="neg_root_mean_squared_error",
                                       n_jobs=nprocs, verbose=verbose)

            grid_search.fit(X_scaled_train, y_scaled_train)

            print("Tuning successful!")

            # Define ML model with tuned hyperparameters
            if model_label == "KN":
                model = KNeighborsRegressor(
                    n_neighbors=grid_search.best_params_["n_neighbors"],
                    weights=grid_search.best_params_["weights"]
                )

            elif model_label == "RF":
                model = RandomForestRegressor(
                    random_state=seed,
                    n_estimators=grid_search.best_params_["n_estimators"],
                    max_features=grid_search.best_params_["max_features"],
                    min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                    min_samples_split=grid_search.best_params_["min_samples_split"]
                )

            elif model_label == "DT":
                model = DecisionTreeRegressor(
                    random_state=seed,
                    splitter=grid_search.best_params_["splitter"],
                    max_features=grid_search.best_params_["max_features"],
                    min_samples_leaf=grid_search.best_params_["min_samples_leaf"],
                    min_samples_split=grid_search.best_params_["min_samples_split"]
                )

            elif model_label in ["NN1", "NN2", "NN3"]:
                model = MLPRegressor(
                    random_state=seed,
                    learning_rate_init=grid_search.best_params_["learning_rate_init"],
                    hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"]
                )

            self.ml_model_tuned = True

        else:
            # Define ML models without tuning
            if model_label == "KN":
                model = KNeighborsRegressor(n_neighbors=4, weights="distance")

            elif model_label == "RF":
                model = RandomForestRegressor(random_state=seed, n_estimators=400,
                                              max_features=2, min_samples_leaf=1,
                                              min_samples_split=2)

            elif model_label == "DT":
                model = DecisionTreeRegressor(random_state=seed, splitter="best",
                                              max_features=2, min_samples_leaf=1,
                                              min_samples_split=2)

            elif model_label == "NN1":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     learning_rate_init=0.001,
                                     hidden_layer_sizes=(nn_L1))

            elif model_label == "NN2":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     learning_rate_init=0.0001,
                                     hidden_layer_sizes=(nn_L3, nn_L2))

            elif model_label == "NN3":
                model = MLPRegressor(random_state=seed, max_iter=epochs,
                                     learning_rate_init=0.0001,
                                     hidden_layer_sizes=(nn_L3, nn_L2, nn_L1))

        # Get trained model
        self.ml_model = model

        # Get hyperparameters
        self.model_hyperparams = model.get_params()

        print("Configuring successful!")

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # iterate kfold !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _iterate_kfold(self, fold_args):
        """
        """
        # Get self attributes
        model = self.ml_model
        model_label = self.ml_model_label
        feature_train = self.feature_train
        target_train = self.target_train
        feature_valid = self.feature_valid
        target_valid = self.target_valid
        epochs = self.epochs
        batchprop = self.batchprop
        fig_dir = self.fig_dir
        verbose = self.verbose

        # Check for training features
        if feature_train.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_train.size == 0:
            raise Exception("No training targets!")

        # Check for validation features
        if feature_valid.size == 0:
            raise Exception("No validation features!")

        # Check for validation targets
        if target_valid.size == 0:
            raise Exception("No validation targets!")

        # Scale training dataset
        X_train, y_train, scaler_X_train, scaler_y_train, X_scaled_train, y_scaled_train = \
            self._scale_arrays(feature_train, target_train)

        # Scale validation dataset
        X_valid, y_valid, scaler_X_valid, scaler_y_valid, X_scaled_valid, y_scaled_valid = \
            self._scale_arrays(feature_valid, target_valid)

        # Get fold indices
        (train_index, test_index) = fold_args

        # Split the data into training and testing sets
        X_train, X_test = X_scaled_train[train_index], X_scaled_train[test_index]
        y_train, y_test = y_scaled_train[train_index], y_scaled_train[test_index]
        X_valid, y_valid = X_scaled_valid, y_scaled_valid

        if "NN" in model_label:
            # Initialize lists to store loss values
            epoch_, train_loss_, valid_loss_ = [], [], []

            # Set batch size as a proportion of the training dataset size
            batch_size = int(len(y_train) * batchprop)

            # Ensure a minimum batch size
            batch_size = max(batch_size, 8)

            # Start training timer
            training_start_time = time.time()

            # Partial training
            with tqdm(total=epochs, desc="Training NN", position=0) as pbar:
                for epoch in range(epochs):
                    # Shuffle the training data for each epoch
                    indices = np.arange(len(y_train))
                    np.random.shuffle(indices)

                    for start_idx in range(0, len(indices), batch_size):
                        end_idx = start_idx + batch_size

                        # Ensure that the batch size doesn't exceed the dataset size
                        end_idx = min(end_idx, len(indices))

                        # Subset training data
                        batch_indices = indices[start_idx:end_idx]
                        X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                        # Train NN model on batch
                        model.partial_fit(X_batch, y_batch)

                    # Calculate and store training loss
                    train_loss = model.loss_
                    train_loss_.append(train_loss)

                    # Calculate and store validation loss
                    valid_loss = mean_squared_error(y_valid, model.predict(X_valid) / 2)
                    valid_loss_.append(valid_loss)

                    # Store epoch
                    epoch_.append(epoch + 1)

                    # Update progress bar
                    pbar.update(1)

            # End training timer
            training_end_time = time.time()

            # Create loss curve dict
            loss_curve = {"epoch": epoch_, "train_loss": train_loss_,
                          "valid_loss": valid_loss_}

        else:
            # Start training timer
            training_start_time = time.time()

            # Train ML model
            model.fit(X_train, y_train)

            # End training timer
            training_end_time = time.time()

            # Empty loss curve
            loss_curve = None

        # Calculate training time
        training_time = (training_end_time - training_start_time) * 1000

        # Make predictions on the test dataset
        y_pred_scaled = model.predict(X_test)
        y_pred_scaled_valid = model.predict(X_valid)

        # Test inference time on single random PT datapoint from the test dataset
        rand_PT_point = X_test[np.random.choice(X_test.shape[0], 1, replace=False)]

        inference_start_time = time.time()
        single_PT_pred = model.predict(rand_PT_point)
        inference_end_time = time.time()

        inference_time = (inference_end_time - inference_start_time) * 1000

        # Inverse transform predictions
        y_pred_original = scaler_y_train.inverse_transform(y_pred_scaled)
        y_pred_original_valid = scaler_y_valid.inverse_transform(y_pred_scaled_valid)

        # Inverse transform test dataset
        y_test_original = scaler_y_train.inverse_transform(y_test)
        y_valid_original = scaler_y_valid.inverse_transform(y_valid)

        # Calculate performance metrics to evaluate the model
        rmse_test = np.sqrt(mean_squared_error(y_test_original, y_pred_original,
                                               multioutput="raw_values"))

        rmse_valid = np.sqrt(mean_squared_error(y_valid_original, y_pred_original_valid,
                                              multioutput="raw_values"))

        r2_test = r2_score(y_test_original, y_pred_original, multioutput="raw_values")
        r2_valid = r2_score(y_valid_original, y_pred_original_valid,
                            multioutput="raw_values")

        return (loss_curve, rmse_test, r2_test, rmse_valid, r2_valid, training_time,
                inference_time)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # process_kfold_results !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _process_kfold_results(self, results):
        """
        """
        # Get self attributes
        model_label = self.ml_model_label
        model_label_full = self.ml_model_label_full
        program = self.program
        targets = self.targets
        kfolds = self.kfolds
        fig_dir = self.fig_dir
        verbose = self.verbose
        W = self.feature_train.shape[0]

        # Initialize empty lists for storing performance metrics
        loss_curves = []
        rmse_test_scores = []
        r2_test_scores = []
        rmse_val_scores = []
        r2_val_scores = []
        training_times = []
        inference_times = []

        # Unpack results
        for (loss_curve, rmse_test, r2_test, rmse_val, r2_val, training_time, inference_time
             ) in results:
            loss_curves.append(loss_curve)
            rmse_test_scores.append(rmse_test)
            r2_test_scores.append(r2_test)
            rmse_val_scores.append(rmse_val)
            r2_val_scores.append(r2_val)
            training_times.append(training_time)
            inference_times.append(inference_time)

        if "NN" in model_label:
            # Initialize empty dict for combined loss curves
            merged_curves = {}

            # Merge loss curves
            for curve in loss_curves:
                for key, value in curve.items():
                    if key in merged_curves:
                        if isinstance(merged_curves[key], list):
                            merged_curves[key].extend(value)
                        else:
                            merged_curves[key] = [merged_curves[key], value]
                            merged_curves[key].extend(value)
                    else:
                        merged_curves[key] = value

            # Make dict into pandas df
            df = pd.DataFrame.from_dict(merged_curves, orient="index").transpose()
            df.sort_values(by="epoch", inplace=True)

            # Set plot style and settings
            plt.rcParams["legend.facecolor"] = "0.9"
            plt.rcParams["legend.loc"] = "upper left"
            plt.rcParams["legend.fontsize"] = "small"
            plt.rcParams["legend.frameon"] = "False"
            plt.rcParams["axes.facecolor"] = "0.9"
            plt.rcParams["font.size"] = 12
            plt.rcParams["figure.autolayout"] = "True"
            plt.rcParams["figure.dpi"] = 330
            plt.rcParams["savefig.bbox"] = "tight"

            # Plot loss curve
            fig = plt.figure(figsize=(6.3, 3.54))

            # Colormap
            colormap = plt.cm.get_cmap("tab10")

            plt.plot(df["epoch"], df["train_loss"], label="train loss", color=colormap(0))
            plt.plot(df["epoch"], df["valid_loss"], label="valid loss", color=colormap(1))
            plt.xlabel("Epoch")
            plt.ylabel(f"Loss")

            if program == "magemin":
                program_label = "MAGEMin"

            if program == "perplex":
                program_label = "Perple_X"

            plt.title(f"{model_label_full} Loss Curve [{program_label}]")
            plt.legend()

            # Save the plot to a file if a filename is provided
            plt.savefig(f"{fig_dir}/{program[:4]}-{model_label}-loss-curve.png")

            # Close plot
            plt.close()

        # Stack arrays
        rmse_test_scores = np.stack(rmse_test_scores)
        r2_test_scores = np.stack(r2_test_scores)
        rmse_val_scores = np.stack(rmse_val_scores)
        r2_val_scores = np.stack(r2_val_scores)

        # Calculate performance values with uncertainties
        rmse_test_mean = np.mean(rmse_test_scores, axis=0)
        rmse_test_std = np.std(rmse_test_scores, axis=0)
        r2_test_mean = np.mean(r2_test_scores, axis=0)
        r2_test_std = np.std(r2_test_scores, axis=0)
        rmse_val_mean = np.mean(rmse_val_scores, axis=0)
        rmse_val_std = np.std(rmse_val_scores, axis=0)
        r2_val_mean = np.mean(r2_val_scores, axis=0)
        r2_val_std = np.std(r2_val_scores, axis=0)
        training_time_mean = np.mean(training_times)
        training_time_std = np.std(training_times)
        inference_time_mean = np.mean(inference_times)
        inference_time_std = np.std(inference_times)

        # Config and performance info
        cv_info = {
            "model": model_label,
            "program": program,
            "size": W,
            "n_targets": len(targets),
            "k_folds": kfolds,
            "training_time_mean": round(training_time_mean, 3),
            "training_time_std": round(training_time_std, 3),
            "inference_time_mean": round(inference_time_mean, 3),
            "inference_time_std": round(inference_time_std, 3)
        }

        # Add performance metrics for each parameter to the dictionary
        for i, target in enumerate(targets):
            cv_info[f"rmse_test_mean_{target}"] = round(rmse_test_mean[i], 3)
            cv_info[f"rmse_test_std_{target}"] = round(rmse_test_std[i], 3)
            cv_info[f"r2_test_mean_{target}"] = round(r2_test_mean[i], 3),
            cv_info[f"r2_test_std_{target}"] = round(r2_test_std[i], 3),
            cv_info[f"rmse_val_mean_{target}"] = round(rmse_val_mean[i], 3),
            cv_info[f"rmse_val_std_{target}"] = round(rmse_val_std[i], 3),
            cv_info[f"r2_val_mean_{target}"] = round(r2_val_mean[i], 3),
            cv_info[f"r2_val_std_{target}"] = round(r2_val_std[i], 3)

        if verbose >= 1:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # Print performance
            print(f"{model_label_full} performance:")
            print(f"    training time: {training_time_mean:.3f} ± {training_time_std:.3f}")
            print(f"    inference time: {inference_time_mean:.3f} ± "
                  f"{inference_time_std:.3f}")
            print(f"    rmse test:")
            for r, e, p in zip(rmse_test_mean, rmse_test_std, targets):
                print(f"        {p}: {r:.3f} ± {e:.3f}")
            print(f"    r2 test:")
            for r, e, p in zip(r2_test_mean, r2_test_std, targets):
                print(f"        {p}: {r:.3f} ± {e:.3f}")
            print(f"    rmse valid:")
            for r, e, p in zip(rmse_val_mean, rmse_val_std, targets):
                print(f"        {p}: {r:.3f} ± {e:.3f}")
            print(f"    r2 valid:")
            for r, e, p in zip(r2_val_mean, r2_val_std, targets):
                print(f"        {p}: {r:.3f} ± {e:.3f}")
            print("+++++++++++++++++++++++++++++++++++++++++++++")

        self.cv_info = cv_info

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # kfold cv !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _kfold_cv(self):
        """
        """
        # Get self attributes
        feature_train = self.feature_train
        target_train = self.target_train
        kfolds = self.kfolds
        nprocs = self.nprocs
        seed = self.seed

        # Check for training features
        if feature_train.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_train.size == 0:
            raise Exception("No training targets!")

        # Scale training dataset
        X_train, y_train, scaler_X_train, scaler_y_train, X_scaled_train, y_scaled_train = \
            self._scale_arrays(feature_train, target_train)

        # Check for ml model
        if self.ml_model is None:
            raise Exception("No ML model! Call _configure_ml_model() first ...")

        # K-fold cross validation
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)

        # Create list of args for mp pooling
        fold_args = [(train_index, test_index) for _, (train_index, test_index) in
                     enumerate(kf.split(X_train))]

        # Create a multiprocessing pool
        with mp.Pool(processes=nprocs) as pool:
            results = pool.map(self._iterate_kfold, fold_args)

            # Wait for all processes
            pool.close()
            pool.join()

        print("Kfold cross validation successful!")
        self.ml_model_cross_validated = True

        # Process cross validation results
        self._process_kfold_results(results)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # retrain !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _retrain(self):
        """
        """
        # Get self attributes
        model = self.ml_model
        model_label = self.ml_model_label
        model_prefix = self.model_prefix
        model_out_dir = self.model_out_dir
        targets = self.targets
        feature_array = self.feature_train_unmasked.copy()
        target_array = self.target_train_unmasked.copy()
        shape_feature_square = self.shape_feature_square
        shape_target_square = self.shape_target_square
        geotherm_mask = self.geotherm_mask_train
        mask_geotherm = self.mask_geotherm
        epochs = self.epochs
        batchprop = self.batchprop
        seed = self.seed
        verbose = self.verbose

        # Check for ml model
        if self.ml_model is None:
            raise Exception("No ML model! Call _configure_ml_model() first ...")

        # Check for ml model cross validated
        if not self.ml_model_cross_validated:
            raise Exception("ML model not cross validated! Call _kfold_cv() first ...")

        # Check for training features
        if feature_array.size == 0:
            raise Exception("No training features!")

        # Check for training targets
        if target_array.size == 0:
            raise Exception("No training targets!")

        print(f"Retraining model {model_prefix} ...")

        # Scale unmasked arrays
        X, y, scaler_X, scaler_y, X_scaled, y_scaled = \
            self._scale_arrays(feature_array, target_array)

        # Train model on entire (unmasked) training dataset
        X_train, X_test, y_train, y_test = \
            train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=seed)

        # Train ML model
        if "NN" in model_label:
            # Set batch size as a proportion of the training dataset size
            batch_size = int(len(y_train) * batchprop)

            # Ensure a minimum batch size
            batch_size = max(batch_size, 8)

            # Partial training
            with tqdm(total=epochs, desc="Retraining NN", position=0) as pbar:
                for epoch in range(epochs):
                    # Shuffle the training data for each epoch
                    indices = np.arange(len(y_train))
                    np.random.shuffle(indices)

                    for start_idx in range(0, len(indices), batch_size):
                        end_idx = start_idx + batch_size

                        # Ensure that the batch size doesn't exceed the dataset size
                        end_idx = min(end_idx, len(indices))

                        # Subset training data
                        batch_indices = indices[start_idx:end_idx]
                        X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]

                        # Train NN model on batch
                        model.partial_fit(X_batch, y_batch)

                    # Update progress bar
                    pbar.update(1)

        else:
            model.fit(X_train, y_train)

        print("Retraining successful!")
        self.ml_model_trained = True
        self.ml_model_only = model

        # Copy feature and target arrays
        X = feature_array.copy()
        y = target_array.copy()

        # Make predictions on unmasked features
        pred_scaled = model.predict(scaler_X.fit_transform(X))

        # Inverse transform predictions
        pred_original = scaler_y.inverse_transform(pred_scaled)

        # Mask geotherm
        if mask_geotherm:
            if verbose >= 2:
                print("Masking geotherm!")

            # Apply mask to all target arrays
            for array in [X, y, pred_original]:
                for j in range(array.shape[-1]):
                    array[:, j][geotherm_mask] = np.nan

        # Reshape arrays into squares for visualization
        feature_square = X.reshape(shape_feature_square)
        target_square = y.reshape(shape_target_square)
        pred_square = pred_original.reshape(shape_target_square)

        # Update arrays
        self.feature_square = feature_square
        self.target_square = target_square
        self.prediction_square = pred_square

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # append to csv !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _append_to_csv(self):
        """
        """
        # Get self attributes
        data_dict = self.cv_info

        # Check for cross validation results
        if not self.cv_info:
            raise Exception("No cross validation! Call _kfold_cv() first ...")

        # CSV filepath
        filepath = f"assets/data/rocmlm-performance.csv"

        # Check if the CSV file already exists
        if not pd.io.common.file_exists(filepath):
            df = pd.DataFrame(data_dict)

        else:
            df = pd.read_csv(filepath)

            # Append the new data dictionary to the DataFrame
            new_data = pd.DataFrame(data_dict)

            df = pd.concat([df, new_data], ignore_index=True)

        # Sort df
        df = df.sort_values(by=["model", "program", "size"])

        # Save the updated DataFrame back to the CSV file
        df.to_csv(filepath, index=False)

        return None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # train ml model !!
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def train(self):
        """
        """
        try:
            # Configure ml model
            self._configure_ml_model()

            if self.verbose >= 1:
                feat_train = self.feature_train_unmasked
                target_train = self.target_train_unmasked

                # Print rocmlm config
                print("+++++++++++++++++++++++++++++++++++++++++++++")
                print("RocMLM model defined as:")
                print(f"    program: {self.program}")
                print(f"    model: {self.ml_model_label_full}")
                if "NN" in self.ml_model_label:
                    print(f"    epochs: {self.epochs}")
                print(f"    k folds: {self.kfolds}")
                print(f"    targets: {self.targets}")
                print(f"    features array shape: {feat_train.shape}")
                print(f"    targets array shape: {target_train.shape}")
                print(f"    hyperparameters:")
                for key, value in self.model_hyperparams.items():
                    print(f"        {key}: {value}")
                print("+++++++++++++++++++++++++++++++++++++++++++++")
                print(f"Running kfold ({self.kfolds}) cross validation ...")

            # Run kfold cross validation
            self._kfold_cv()

            # Retrain ml model on unmasked training dataset
            self._retrain()

            # Save ML model performance info to csv
            self._append_to_csv()

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            return None

        except Exception as e:
            print(f"Error occurred during RocMLM training for {self.model_prefix}!")
            traceback.print_exc()

            self.ml_model_training_error = True
            self.ml_model_error = e

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            return None