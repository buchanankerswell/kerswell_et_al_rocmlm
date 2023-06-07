# Magemin program
MAGEMIN = MAGEMin/MAGEMin

# Python scripts
PYTHON = python/conda-specs.txt python/magemin.py python/clone-magemin.py python/build-upper-mantle-database.py python/visualize-upper-mantle-database.py

# Logging
LOGFILE := log/log-$(shell date +%Y-%m-%d)

# Use bash shell to nsure that make stops if commands fail before piped tee
SHELL = /bin/bash -o pipefail

# Conda environment variables
HAS_CONDA := $(shell command -v conda >/dev/null && echo true || echo false)
CONDA_ENV_NAME = madnn
CONDA_ENV_DIR=$(shell conda info --base)
MY_ENV_DIR=$(CONDA_ENV_DIR)/envs/$(CONDA_ENV_NAME)
CONDA_SPECS_FILE = python/conda-specs.txt
CONDA_PYTHON = $$(conda run -n $(CONDA_ENV_NAME) which python)

# Github repos
MAGEMIN_REPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
GITHUB_REPO = https://github.com/buchanankerswell/kerswell_et_al_madnn

# Cleanup
DATAPURGE = python/__pycache__
FIGSPURGE = figs
DATACLEAN = runs log MAGEMin

all: create_conda_env $(PYTHON) $(LOGFILE) $(MAGEMIN)
	@echo "Building MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/build-upper-mantle-database.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes > /dev/null 2>&1
	@echo "Done!"

visualisize: create_conda_env $(PYTHON) $(LOGFILE)
	@echo "Visualizing MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-upper-mantle-database.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes > /dev/null 2>&1
	@echo "Done!"

$(MAGEMIN): $(PYTHON) $(LOGFILE)
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
		conda create --name $(CONDA_ENV_NAME) --file $(CONDA_SPECS_FILE) --yes > /dev/null 2>&1; \
		conda_channel=$$(conda config --get channels); \
		if [[ "$$conda_channel" != *"conda-forge"* ]]; then \
		  echo "conda-forge channel not found. ..." 2>&1 | tee -a $(LOGFILE); \
			echo "Setting conda-forge as highest-priority channel ..." 2>&1 | tee -a $(LOGFILE); \
			conda config --add channels conda-forge; \
	  fi; \
	fi

$(LOGFILE):
	@mkdir log
	@touch $(LOGFILE)

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN)

.PHONY: all create_conda_env purge clean purge