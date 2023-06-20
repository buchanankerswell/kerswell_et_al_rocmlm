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
DATA = assets/data
CONFIG = assets/config
PYTHON = python/conda-environment.yaml \
				 python/magemin.py \
				 python/clone-magemin.py \
				 python/build-database.py \
				 python/submit-jobs.py \
				 python/visualize-database.py \
				 python/download-assets.py
SEED = 32
PRANGE ?= [10, 60, 10]
TRANGE ?= [500, 2500, 400]
SOURCE ?= earthchem
STRATEGY ?= batch
N ?= 3
K ?= 1
PARALLEL ?= True
NPROCS ?= $(shell expr $(shell nproc) - 2)
OUTDIR ?= $(shell pwd)/runs
FIGDIR ?= $(shell pwd)/figs
OXIDES ?= '["MgO", "FeO", "CaO", "Al2O3"]'
PARAMS ?= '[ \
					"Status", \
					"Vp", \
					"Vs", \
					"StableSolutions", \
					"LiquidFraction", \
					"DensityOfFullAssemblage", \
					"DensityOfLiquid", \
					"DensityOfSolid", \
					]'
DATAPURGE = python/__pycache__ .job output
DATACLEAN = assets log MAGEMin runs
FIGSPURGE = figs
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env $(DATA) $(CONFIG) $(MAGEMIN)
	@$(CONDA_PYTHON) python/session-info.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)
	@echo "Run any of the following:" 2>&1 | tee -a $(LOGFILE)
	@echo -e \
	"make build_database\n\
  PRANGE=   <\"[from, to, by]\"> in kbar\n\
  TRANGE=   <\"[from, to, by]\"> in celcius\n\
  SOURCE=   <earthchem>\n\
  STRATEGY= <batch or random>\n\
  N=        <number of samples>\n\
  K=        <batch number>\n\
  PARALLEL= <True or False>\n\
  NPROCS=   <number of processors for parallel>\n\
  SEED=     <number for random state>\n\
  OUTDIR=   <directory of MAGEMin output>" \
	2>&1 | tee -a $(LOGFILE)
	@echo -e \
	"make visualize_database\n\
  PARAMS=   <\'[\"param\", \"param\", \"param\"]\'>\n\
    Options:\n\
    Point,Status, Gibbs, BrNorm, Vp, Vs, Entropy, StableSolutions\n\
    LiquidFraction, DensityOfFullAssemblage, DensityOfLiquid, DensityOfSolid\n\
    DensityOfMixture\n\
  OXIDES=   <\'[\"oxide\", \"oxide\", \"oxide\"]\'> list of options below\n\
    Options:\n\
    SiO2, Al2O3, CaO, MgO, FeO, K2O, Na2O, TiO2, Fe2O3, Cr2O3, H2O\n\
  OUTDIR=   <directory of MAGEMin output>\n\
  FIGDIR=   <directory for saving plots>" \
	2>&1 | tee -a $(LOGFILE)
	@echo "make submit_jobs" 2>&1 | tee -a $(LOGFILE)
	@echo "make remove_conda_env" 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

visualize_database: $(LOGFILE) $(PYTHON)
	@echo "Visualizing MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-database.py \
	--params=$(PARAMS) \
	--oxides=$(OXIDES) \
	--outdir=$(OUTDIR) \
	--figdir=$(FIGDIR) \
	2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

submit_jobs: $(LOGFILE) $(PYTHON) $(DATA)
	@echo "Submitting job to SLURM ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/submit-jobs.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

build_database: $(LOGFILE) $(PYTHON) $(DATA) $(CONFIG) $(MAGEMIN)
	@echo "Building MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/build-database.py \
	--Prange "$(PRANGE)" \
	--Trange "$(TRANGE)" \
	--source $(SOURCE) \
	--strategy $(STRATEGY) \
	--n $(N) \
	--k $(K) \
	--parallel $(PARALLEL) \
	--nprocs $(NPROCS) \
	--seed $(SEED) \
	--outdir $(OUTDIR) \
	2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

$(MAGEMIN): $(LOGFILE) $(PYTHON) $(CONFIG)
	@if [ ! -e "$(MAGEMIN)" ]; then \
	  echo "Cloning MAGEMin from $(MAGEMIN_REPO) ..." 2>&1 | tee -a $(LOGFILE); \
	  chmod +x python/clone-magemin.py; \
	  $(CONDA_PYTHON) python/clone-magemin.py 2>&1 | tee -a $(LOGFILE); \
	  if [ "$(UNAME_S)" = "Darwin" ]; then \
	    echo "Configuring MAGEMin for apple ..." 2>&1 | tee -a $(LOGFILE); \
	    cp $(CONFIG)/PGE_function.c MAGEMin/src/; \
	    cp $(CONFIG)/MAGEMin-apple MAGEMin/Makefile; \
	  fi; \
	  if [ "$(UNAME_S)" = "Linux" ]; then \
	    echo "Configuring MAGEMin for meso ..." 2>&1 | tee -a $(LOGFILE); \
	    cp $(CONFIG)/MAGEMin-meso MAGEMin/Makefile; \
	  fi; \
	  echo "Compiling MAGEMin ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 2>&1 | tee -a $(LOGFILE); \
	  (cd MAGEMin && make) 2>&1 | tee -a $(LOGFILE); \
	  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 2>&1 | tee -a $(LOGFILE); \
	else \
	  echo "MAGEMin found ..." 2>&1 | tee -a $(LOGFILE); \
	fi
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

remove_conda_env: $(LOGFILE)
	@echo "Removing conda env $(CONDA_ENV_NAME) ..." 2>&1 | tee -a $(LOGFILE)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

create_conda_env: $(LOGFILE) $(CONDA_SPECS_FILE) find_conda_env
	@if [ "$(HAS_CONDA)" = "false" ]; then \
	  echo "conda not found in PATH, install conda first ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "See $(GITHUB_REPO) for details ..." 2>&1 | tee -a $(LOGFILE); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
		echo "Env \"$(CONDA_ENV_NAME)\" found in:"; \
	  echo "$(MY_ENV_DIR)"; \
		  2>&1 | tee -a $(LOGFILE); \
	else \
		echo "Creating env $(CONDA_ENV_NAME) from $(CONDA_SPECS_FILE) ..." \
		  2>&1 | tee -a $(LOGFILE); \
		conda env create --file $(CONDA_SPECS_FILE) > /dev/null \
		  2>&1 | tee -a $(LOGFILE); \
	  echo "Conda env created ..." 2>&1 | tee -a $(LOGFILE); \
	fi
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

find_conda_env:
	@echo "Looking for conda env ..." 2>&1 | tee -a $(LOGFILE)
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDA_ENV_NAME) | awk '{print $$2}'))

$(DATA): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py 2>&1 | tee -a $(LOGFILE); \
	else \
		echo "$(DATA) found..." 2>&1 | tee -a $(LOGFILE); \
	fi
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

$(CONFIG): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(CONFIG)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py 2>&1 | tee -a $(LOGFILE); \
	else \
		echo "$(CONFIG) found..." 2>&1 | tee -a $(LOGFILE); \
	fi
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

$(LOGFILE):
	@mkdir log
	@touch $(LOGFILE)

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN) $(FIGSCLEAN)

.PHONY: find_conda_env remove_conda_env create_conda_env build_database submit_jobs visualize_earthchem visualize_database all purge clean