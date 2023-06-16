SHELL = /bin/bash -o pipefail
UNAME_S := $(shell uname -s)
LOGFILE := log/log-$(shell date +%Y-%m-%d)
HAS_CONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDA_ENV_NAME = madnn
CONDA_SPECS_FILE = python/conda-environment.yaml
CONDA_PYTHON = $$(conda run -n $(CONDA_ENV_NAME) which python)
MAGEMIN = MAGEMin/MAGEMin
MAGEMIN_REPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
GITHUB_REPO = https://github.com/buchanankerswell/kerswell_et_al_madnn
DATA = data
PYTHON = python/conda-environment.yaml \
				 python/magemin.py \
				 python/clone-magemin.py \
				 python/build-database.py \
				 python/visualize-database.py \
				 python/download-data.py \
				 python/visualize-earthchem.py
DATAPURGE = python/__pycache__
DATACLEAN = data runs output log MAGEMin
FIGSPURGE =
FIGSCLEAN = figs

all: $(LOGFILE) create_conda_env
	@echo "Now you can run any of the following:" 2>&1 | tee -a $(LOGFILE)
	@echo "make build_database" 2>&1 | tee -a $(LOGFILE)
	@echo "make visualize_earthchem" 2>&1 | tee -a $(LOGFILE)
	@echo "make visualize_databse" 2>&1 | tee -a $(LOGFILE)
	@echo "make remove_conda_env" 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

visualize_earthchem: $(LOGFILE) $(PYTHON) $(DATA)
	@echo "Visualizing Earthchem database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-earthchem.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

visualize_database: $(LOGFILE) $(PYTHON) $(DATA)
	@echo "Visualizing MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-database.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

build_database: $(LOGFILE) $(PYTHON) $(DATA) $(MAGEMIN)
	@if [ "$(UNAME_S)" = "Darwin" ]; then \
	  echo "Building MAGEMin database ..." 2>&1 | tee -a $(LOGFILE); \
	  $(CONDA_PYTHON) python/build-database.py 2>&1 | tee -a $(LOGFILE); \
	fi
	@if [ "$(UNAME_S)" = "Linux" ]; then \
	  echo "Submitting job to SLURM ..." 2>&1 | tee -a $(LOGFILE); \
	fi
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

$(MAGEMIN): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(MAGEMIN)" ]; then \
	  echo "Cloning MAGEMin from $(MAGEMIN_REPO) ..." 2>&1 | tee -a $(LOGFILE); \
	  chmod +x python/clone-magemin.py; \
	  $(CONDA_PYTHON) python/clone-magemin.py 2>&1 | tee -a $(LOGFILE); \
	  if [ "$(UNAME_S)" = "Darwin" ]; then \
	    echo "Configuring MAGEMin for apple ..." 2>&1 | tee -a $(LOGFILE); \
	    cp config/PGE_function.c MAGEMin/src/; \
	    cp config/MAGEMin-apple MAGEMin/Makefile; \
	  fi; \
	  if [ "$(UNAME_S)" = "Linux" ]; then \
	    echo "Configuring MAGEMin for meso ..." 2>&1 | tee -a $(LOGFILE); \
	    cp config/MAGEMin-meso MAGEMin/Makefile; \
	  fi; \
	  echo "Compiling MAGEMin ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
	  (cd MAGEMin && make) 2>&1 | tee -a $(LOGFILE); \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
	else \
	  echo "MAGEMin found ..." 2>&1 | tee -a $(LOGFILE); \
	fi

remove_conda_env: $(LOGFILE)
	@echo "Removing conda env $(CONDA_ENV_NAME) ..." 2>&1 | tee -a $(LOGFILE)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

create_conda_env: $(LOGFILE) $(CONDA_SPECS_FILE) find_conda_env
	@if [ "$(HAS_CONDA)" = "false" ]; then \
	  echo "conda not found in PATH, install conda first ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "See $(GITHUB_REPO) for details ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
		echo "Detected conda ..." 2>&1 | tee -a $(LOGFILE); \
		echo "Found environment \"$(CONDA_ENV_NAME)\" in $(MY_ENV_DIR) ..." \
		  2>&1 | tee -a $(LOGFILE); \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
	else \
		echo "Detected conda ..." 2>&1 | tee -a $(LOGFILE); \
		echo "Creating environment \"$(CONDA_ENV_NAME)\" from $(CONDA_SPECS_FILE) ..." \
		  2>&1 | tee -a $(LOGFILE); \
		conda env create --file $(CONDA_SPECS_FILE) > /dev/null \
		  2>&1 | tee -a $(LOGFILE); \
	  echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
	  echo "Conda env created ..." 2>&1 | tee -a $(LOGFILE); \
	fi

find_conda_env:
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDA_ENV_NAME) | awk '{print $$2}'))

$(DATA): $(LOGFILE) $(PYTHON)
	@$(CONDA_PYTHON) python/download-data.py 2>&1 | tee -a $(LOGFILE)

$(LOGFILE):
	@mkdir log
	@touch $(LOGFILE)

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN) $(FIGSCLEAN)

.PHONY: find_conda_env remove_conda_env create_conda_env build_database visualize_earthchem visualize_database all purge clean