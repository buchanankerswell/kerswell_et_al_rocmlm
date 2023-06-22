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
PERPLEX = assets/perplex
DATA = assets/data
CONFIG = assets/MAGEMin-meso-config
PYTHON = python/conda-environment.yaml \
				 python/magemin.py \
				 python/clone-magemin.py \
				 python/build-database.py \
				 python/benchmark-magemin-perplex.py \
				 python/submit-jobs.py \
				 python/visualize-database.py \
				 python/download-assets.py
SEED = 32
PMIN ?= 10
PMAX ?= 110
PRES ?= 128
TMIN ?= 500
TMAX ?= 2500
TRES ?= 128
COMP ?= [44.9, 4.44, 3.54, 37.71, 8.03, 0.029, 0.36, 0.2, 0.01, 0.38, 0]
FRAC ?= wt
SAMPLEID ?= PUM
NORMOX ?= all
SOURCE ?= earthchem
STRATEGY ?= random
N ?= 1
K ?= 0
PARALLEL ?= True
NPROCS ?= $(shell expr $(shell nproc) - 2)
OUTDIR ?= runs
FIGDIR ?= figs
FIGOX ?= '["MgO", "FeO", "CaO", "Al2O3"]'
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

all: $(LOGFILE) $(PYTHON) create_conda_env $(DATA) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/session-info.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)
	@echo "Run any of the following:" 2>&1 | tee -a $(LOGFILE)
	@echo -e \
	"make build_database\nmake benchmark-magemin-perplex\n\
  PMIN=     <kbar>\n\
  PMAX=     <kbar>\n\
  PRES=     <number of points>\n\
  TMIN=     <Celcius>\n\
  TMAX=     <Celcius>\n\
  TRES=     <number of points>\n\
  COMP=     <'[SiO2, Al2O3, CaO, MgO, FeO, K2O, Na2O, TiO2, Fe2O3, Cr2O3, H2O]'>\n\
  FRAC=     <mol or wt>\n\
  SAMPLEID= <sample name>\n\
  NORMOX=   <'[\"oxide\", \"oxide\", \"oxide\"]'> or <all>\n\
  SOURCE=   <earthchem or sample>\n\
  STRATEGY= <batch or random>\n\
  N=        <number of samples>\n\
  K=        <batch number>\n\
  PARALLEL= <True or False>\n\
  NPROCS=   <number of processors for parallel>\n\
  SEED=     <number for random state>\n\
  OUTDIR=   <directory of MAGEMin output>" \
	2>&1 | tee -a $(LOGFILE)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 2>&1 | tee -a $(LOGFILE)
	@echo -e \
	"make visualize_database\n\
  PARAMS=   <'[\"param\", \"param\", \"param\"]'>\n\
    Options:\n\
    Point,Status, Gibbs, BrNorm, Vp, Vs, Entropy, StableSolutions\n\
    LiquidFraction, DensityOfFullAssemblage, DensityOfLiquid, DensityOfSolid\n\
    DensityOfMixture\n\
  FIGOX=   <'[\"oxide\", \"oxide\", \"oxide\"]'> for Harker diagrams\n\
    Options:\n\
    SiO2, Al2O3, CaO, MgO, FeO, K2O, Na2O, TiO2, Fe2O3, Cr2O3, H2O\n\
  OUTDIR=   <directory of MAGEMin output>\n\
  FIGDIR=   <directory for saving plots>" \
	2>&1 | tee -a $(LOGFILE)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 2>&1 | tee -a $(LOGFILE)
	@echo "make submit_jobs" 2>&1 | tee -a $(LOGFILE)
	@echo "make remove_conda_env" 2>&1 | tee -a $(LOGFILE)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

visualize_database: $(LOGFILE) $(PYTHON)
	@echo "Visualizing MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/visualize-database.py \
	--params=$(PARAMS) \
	--figox=$(FIGOX) \
	--outdir=$(OUTDIR) \
	--figdir=$(FIGDIR) \
	2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

submit_jobs: $(LOGFILE) $(PYTHON) $(DATA)
	@echo "Submitting job to SLURM ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/submit-jobs.py 2>&1 | tee -a $(LOGFILE)
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

benchmark_magemin_perplex: $(LOGFILE) $(PYTHON) $(DATA) $(PERPLEX) $(MAGEMIN)
	@echo "Building MAGEMin model ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/benchmark-magemin-perplex.py \
	--Pmin $(PMIN) \
	--Pmax $(PMAX) \
	--Pres $(PRES) \
	--Tmin $(TMIN) \
	--Tmax $(TMAX) \
	--Tres $(TRES) \
	--comp '$(COMP)' \
	--frac $(FRAC) \
	--sampleid $(SAMPLEID) \
	--normox '$(NORMOX)' \
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
	@chmod +x $(PERPLEX)/build $(PERPLEX)/vertex $(PERPLEX)/pssect
	@if [ ! -e "$(PERPLEX)/$(SAMPLEID).dat" ]; then \
	  echo "Building Perple_x model ..." 2>&1 | tee -a $(LOGFILE); \
		(cd $(PERPLEX) && ./build < perplex-benchmark-$(SAMPLEID)) 2>&1 | tee -a $(LOGFILE); \
	  (cd $(PERPLEX) && echo "$(SAMPLEID)" | ./vertex) 2>&1 | tee -a $(LOGFILE); \
	fi
	@if [ -e "$(PERPLEX)/$(SAMPLEID).dat" ]; then \
	  echo "Plotting Perple_x model ..." 2>&1 | tee -a $(LOGFILE); \
		echo "$(SAMPLEID)" > $(PERPLEX)/$(SAMPLEID)-pssect-config; \
		echo "N" >> $(PERPLEX)/$(SAMPLEID)-pssect-config; \
	  (cd $(PERPLEX) && ./pssect < $(SAMPLEID)-pssect-config) 2>&1 | tee -a $(LOGFILE); \
		(cd $(PERPLEX) && ps2pdf $(SAMPLEID).ps $(SAMPLEID).pdf) 2>&1 | tee -a $(LOGFILE); \
		rm $(PERPLEX)/$(SAMPLEID)-pssect-config; \
	fi
	@echo "=============================================" 2>&1 | tee -a $(LOGFILE)

build_database: $(LOGFILE) $(PYTHON) $(DATA) $(MAGEMIN)
	@echo "Building MAGEMin database ..." 2>&1 | tee -a $(LOGFILE)
	@$(CONDA_PYTHON) python/build-database.py \
	--Pmin $(PMIN) \
	--Pmax $(PMAX) \
	--Pres $(PRES) \
	--Tmin $(TMIN) \
	--Tmax $(TMAX) \
	--Tres $(TRES) \
	--comp '$(COMP)' \
	--frac $(FRAC) \
	--sampleid $(SAMPLEID) \
	--normox '$(NORMOX)' \
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
		echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
	  echo "Cloning MAGEMin from $(MAGEMIN_REPO) ..." 2>&1 | tee -a $(LOGFILE); \
	  chmod +x python/clone-magemin.py; \
	  $(CONDA_PYTHON) python/clone-magemin.py 2>&1 | tee -a $(LOGFILE); \
	  if [ "$(UNAME_S)" = "Linux" ]; then \
	    echo "Configuring MAGEMin for meso ..." 2>&1 | tee -a $(LOGFILE); \
	    cp $(CONFIG) MAGEMin/Makefile; \
	  fi; \
	  echo "Compiling MAGEMin ..." 2>&1 | tee -a $(LOGFILE); \
	  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 2>&1 | tee -a $(LOGFILE); \
	  (cd MAGEMin && make) 2>&1 | tee -a $(LOGFILE); \
	  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 2>&1 | tee -a $(LOGFILE); \
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
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
		echo "Env \"$(CONDA_ENV_NAME)\" found in: $(MY_ENV_DIR) ..." 2>&1 | tee -a $(LOGFILE); \
	else \
		echo "Creating env $(CONDA_ENV_NAME) from $(CONDA_SPECS_FILE) ..." \
		  2>&1 | tee -a $(LOGFILE); \
		conda env create --file $(CONDA_SPECS_FILE) > /dev/null \
		  2>&1 | tee -a $(LOGFILE); \
	  echo "Conda env created ..." 2>&1 | tee -a $(LOGFILE); \
	fi

find_conda_env:
	@echo "Looking for conda env ..." 2>&1 | tee -a $(LOGFILE)
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDA_ENV_NAME) | awk '{print $$2}'))

$(PERPLEX): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py 2>&1 | tee -a $(LOGFILE); \
	else \
		echo "$(PERPLEX) found..." 2>&1 | tee -a $(LOGFILE); \
	fi

$(DATA): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py 2>&1 | tee -a $(LOGFILE); \
	else \
		echo "$(DATA) found..." 2>&1 | tee -a $(LOGFILE); \
	fi

$(CONFIG): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIG)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py 2>&1 | tee -a $(LOGFILE); \
		echo "=============================================" 2>&1 | tee -a $(LOGFILE); \
	else \
		echo "$(CONFIG) found..." 2>&1 | tee -a $(LOGFILE); \
	fi

$(LOGFILE):
	@if [ ! -d "$(LOGFILE)" ]; then \
		mkdir log; \
		touch $(LOGFILE); \
	fi

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN) $(FIGSCLEAN)

.PHONY: find_conda_env remove_conda_env create_conda_env build_database submit_jobs visualize_earthchem visualize_database all purge clean