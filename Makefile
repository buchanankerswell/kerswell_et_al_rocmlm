# Magemin program
MAGEMIN = MAGEMin/MAGEMin

# Python scripts
PYTHON = python/conda-environment.yaml python/magemin.py python/clone-magemin.py python/build-upper-mantle-database.py python/visualize-upper-mantle-database.py

# Logging
LOGFILE := log/log-$(shell date +%Y-%m-%d)

# Use bash shell to nsure that make stops if commands fail before piped tee
SHELL = /bin/bash -o pipefail

# Conda environment variables
HAS_CONDA := $(shell command -v conda >/dev/null && echo true || echo false)
CONDA_ENV_NAME = madnn
CONDA_SPECS_FILE = python/conda-environment.yaml

# Search for Conda environment directory
CONDA_ENV_DIR := $(shell \
		if [ -d "$(shell conda info --base)/envs/$(CONDA_ENV_NAME)" ]; then \
			echo "$(shell conda info --base)/envs"; \
		elif [ -d "$(HOME)/.conda/envs/$(CONDA_ENV_NAME)" ]; then \
			echo "$(HOME)/.conda/envs"; \
		fi \
	)

# Conda environment variables
MY_ENV_DIR=$(CONDA_ENV_DIR)/envs/$(CONDA_ENV_NAME)
CONDA_PYTHON = $$(conda run -n $(CONDA_ENV_NAME) which python)

# Github repos
MAGEMIN_REPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
GITHUB_REPO = https://github.com/buchanankerswell/kerswell_et_al_madnn

# Cleanup
DATAPURGE = python/__pycache__
FIGSPURGE = figs
DATACLEAN = runs log MAGEMin

all: create_conda_env build_database visualize_database $(LOGFILE)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes > /dev/null 2>&1
	@echo "Done!" 2>&1 | tee -a $(LOGFILE)

visualize_database: $(MY_ENV_DIR) $(PYTHON) $(LOGFILE)
	@echo "Visualizing MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-upper-mantle-database.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

build_database: $(MAGEMIN) $(MY_ENV_DIR) $(PYTHON) $(LOGFILE)
	@echo "Building MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/build-upper-mantle-database.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

$(MAGEMIN): $(MY_ENV_DIR) $(PYTHON) $(LOGFILE)
	@echo "Cloning MAGEMin from $(MAGEMIN_REPO) ..." 2>&1 | tee -a $(LOGFILE)
	@chmod +x python/clone-magemin.py
	@$(CONDA_PYTHON) python/clone-magemin.py 2>&1 | tee -a $(LOGFILE)
	@echo "Compiling MAGEMin ..." 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)
	@(cd MAGEMin && make) 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

create_conda_env: $(CONDA_SPECS_FILE) $(LOGFILE)
	@if [ "$(HAS_CONDA)" = "false" ]; then \
	  echo "conda not found in PATH, install conda first ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "See $(GITHUB_REPO) for details ..." 2>&1 | tee -a $(LOGFILE); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
		echo "Found environment \"$(CONDA_ENV_NAME)\" in $(MY_ENV_DIR) ..." \
		  2>&1 | tee -a $(LOGFILE); \
	else \
		echo "Detected conda ..." 2>&1 | tee -a $(LOGFILE); \
		echo "Creating environment \"$(CONDA_ENV_NAME)\" from $(CONDA_SPECS_FILE) ..." \
		  2>&1 | tee -a $(LOGFILE); \
		conda env create --file $(CONDA_SPECS_FILE) \
		  2>&1 | tee -a $(LOGFILE); \
	fi

$(LOGFILE):
	@mkdir log
	@touch $(LOGFILE)

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN)

.PHONY: all create_conda_env purge clean purge