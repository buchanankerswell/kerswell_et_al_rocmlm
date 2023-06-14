SHELL = /bin/bash -o pipefail
LOGFILE := log/log-$(shell date +%Y-%m-%d)
DATA = data
HAS_CONDA := $(shell command -v conda >/dev/null && echo true || echo false)
CONDA_ENV_NAME = madnn
CONDA_SPECS_FILE = python/conda-environment.yaml
CONDA_PYTHON = $$(conda run -n $(CONDA_ENV_NAME) which python)
MAGEMIN_REPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
GITHUB_REPO = https://github.com/buchanankerswell/kerswell_et_al_madnn
MAGEMIN = MAGEMin/MAGEMin
PYTHON = python/conda-environment.yaml python/magemin.py python/clone-magemin.py python/build-upper-mantle-database.py python/visualize-upper-mantle-database.py
DATAPURGE = python/__pycache__
DATACLEAN = data runs output log MAGEMin
FIGSPURGE =
FIGSCLEAN = figs

all: create_conda_env build_database visualize_database $(LOGFILE)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes > /dev/null 2>&1
	@echo "Done!" 2>&1 | tee -a $(LOGFILE)

visualize_database: create_conda_env $(MY_ENV_DIR) $(DATA) $(PYTHON) $(LOGFILE)
	@echo "Visualizing Earthchem database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-earthchem-samples.py 2>&1 | tee -a $(LOGFILE)
	@echo "Visualizing MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-upper-mantle-database.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

build_database: create_conda_env $(MAGEMIN) $(MY_ENV_DIR) $(DATA) $(PYTHON) $(LOGFILE)
	@echo "Building MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/build-upper-mantle-database.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

$(DATA): $(PYTHON) $(LOGFILE)
	@$(CONDA_PYTHON) python/download-data-osf-repo.py 2>&1 | tee -a $(LOGFILE)

$(MAGEMIN): $(MY_ENV_DIR) $(PYTHON) $(LOGFILE)
	@echo "Cloning MAGEMin from $(MAGEMIN_REPO) ..." 2>&1 | tee -a $(LOGFILE)
	@chmod +x python/clone-magemin.py
	@$(CONDA_PYTHON) python/clone-magemin.py 2>&1 | tee -a $(LOGFILE)
	@echo "Configuring MAGEMin ..." 2>&1 | tee -a $(LOGFILE)
	@if [ -z $$(command -v clang) ]; then \
	  echo "clang not found, uncommenting CC=gcc in MAGEMin Makefile ..." \
	    2>&1 | tee -a $(LOGFILE); \
	  sed -i 's/# CC=gcc/CC=gcc/' MAGEMin/Makefile; \
	  sed -i 's/CC=clang/# CC=clang/' MAGEMin/Makefile; \
	fi
	@echo "Compiling MAGEMin ..." 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)
	@(cd MAGEMin && make) 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

create_conda_env: find_conda_env $(CONDA_SPECS_FILE) $(LOGFILE)
	@if [ "$(HAS_CONDA)" = "false" ]; then \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
	  echo "conda not found in PATH, install conda first ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "See $(GITHUB_REPO) for details ..." 2>&1 | tee -a $(LOGFILE); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
		echo "Found environment \"$(CONDA_ENV_NAME)\" in $(MY_ENV_DIR) ..." \
		  2>&1 | tee -a $(LOGFILE); \
	else \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
		echo "Detected conda ..." 2>&1 | tee -a $(LOGFILE); \
		echo "Creating environment \"$(CONDA_ENV_NAME)\" from $(CONDA_SPECS_FILE) ..." \
		  2>&1 | tee -a $(LOGFILE); \
		conda env create --file $(CONDA_SPECS_FILE) \
		  2>&1 | tee -a $(LOGFILE); \
	fi

find_conda_env:
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDA_ENV_NAME) | awk '{print $$2}'))

$(LOGFILE):
	@mkdir log
	@touch $(LOGFILE)

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN) $(FIGSCLEAN)

.PHONY: find_conda_env create_conda_env build_database visualize_database all purge clean