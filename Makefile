SHELL = /bin/bash -o pipefail
WORKDIR = $(shell pwd)
UNAME_S := $(shell uname -s)
LOGFILE := $(WORKDIR)/log/log-$(shell date +%Y-%m-%d)
LOG := 2>&1 | tee -a $(LOGFILE)
HAS_CONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDA_ENV_NAME = madnn
CONDA_SPECS_FILE = $(WORKDIR)/python/conda-environment.yaml
CONDA_PYTHON = $$(conda run -n $(CONDA_ENV_NAME) which python)
MAGEMIN = $(WORKDIR)/MAGEMin/MAGEMin
MAGEMIN_REPO = https://github.com/ComputationalThermodynamics/MAGEMin.git
GITHUB_REPO = https://github.com/buchanankerswell/kerswell_et_al_madnn
PERPLEX = $(WORKDIR)/assets/perplex
BENCHMARK = $(WORKDIR)/assets/benchmark
DATA = $(WORKDIR)/assets/data
CONFIG = $(WORKDIR)/assets/MAGEMin-meso-config
PYTHON = $(WORKDIR)/python/conda-environment.yaml \
				 $(WORKDIR)/python/magemin.py \
				 $(WORKDIR)/python/clone-magemin.py \
				 $(WORKDIR)/python/build-database.py \
				 $(WORKDIR)/python/benchmark-magemin-perplex.py \
				 $(WORKDIR)/python/submit-jobs.py \
				 $(WORKDIR)/python/visualize-database.py \
				 $(WORKDIR)/python/download-assets.py
SEED = 32
PMIN ?= 10
PMAX ?= 110
PRES ?= 40
TMIN ?= 500
TMAX ?= 2500
TRES ?= 40
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
FIGOX ?= ["MgO", "FeO", "CaO", "Al2O3"]
PARAMS ?= ["Vp", "StableSolutions", "StableVariance", "DensityOfFullAssemblage"]
DATAPURGE = python/__pycache__ .job output
DATACLEAN = assets log MAGEMin runs
FIGSPURGE = figs
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env $(DATA) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDA_PYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@echo "Run any of the following:" $(LOG)
	@echo -e \
	"make build_database\nmake benchmark_magemin_perplex\n\
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
  OUTDIR=   <directory of MAGEMin output>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
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
  FIGDIR=   <directory for saving plots>" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo "make submit_jobs" $(LOG)
	@echo "make remove_conda_env" $(LOG)
	@echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG)
	@echo "=============================================" $(LOG)

visualize_database: $(LOGFILE) $(PYTHON)
	@echo "Visualizing MAGEMin database ..." $(LOG)
	@$(CONDA_PYTHON) python/visualize-database.py \
	--params '$(PARAMS)' \
	--figox '$(FIGOX)' \
	--outdir $(OUTDIR) \
	--figdir $(FIGDIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

submit_jobs: $(LOGFILE) $(PYTHON) $(DATA)
	@echo "Submitting job to SLURM ..." $(LOG)
	@$(CONDA_PYTHON) python/submit-jobs.py $(LOG)
	@echo "=============================================" $(LOG)

benchmark_magemin_perplex: $(LOGFILE) $(PYTHON) $(DATA) $(PERPLEX) $(MAGEMIN)
	@echo "Building MAGEMin model ..." $(LOG)
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
	$(LOG)
	@echo "=============================================" $(LOG)
	@chmod +x $(PERPLEX)/build $(PERPLEX)/vertex $(PERPLEX)/pssect
	@if [ ! -e "$(PERPLEX)/$(SAMPLEID).dat" ]; then \
	  echo "Building Perple_x model ..." $(LOG); \
		(cd $(PERPLEX) && ./build < $(BENCHMARK)/$(SAMPLEID)/build-config) $(LOG); \
	  (cd $(PERPLEX) && echo "$(SAMPLEID)" | ./vertex) $(LOG); \
	fi
	@if [ -e "$(PERPLEX)/$(SAMPLEID).dat" ]; then \
	  echo "Plotting Perple_x model ..." $(LOG); \
		(cd $(PERPLEX) && echo "$(SAMPLEID)" > pssect-config) $(LOG); \
		(cd $(PERPLEX) && echo "N" >> pssect-config) $(LOG); \
	  (cd $(PERPLEX) && ./pssect < pssect-config) $(LOG); \
		(cd $(PERPLEX) && ps2pdf $(SAMPLEID).ps $(SAMPLEID).pdf) $(LOG); \
		( \
		cd $(PERPLEX) && \
		convert $(SAMPLEID).pdf -crop 565x495+0+165 -resize 1994x1601 $(SAMPLEID).png \
		) $(LOG); \
		(cd $(PERPLEX) && rm pssect-config $(SAMPLEID).pdf) $(LOG); \
		(cd $(PERPLEX) && mv $(SAMPLEID)* $(BENCHMARK)/$(SAMPLEID)) $(LOG); \
	fi
	@ echo "Finished Perple_x model ..." $(LOG)
	@echo "=============================================" $(LOG)
	@$(MAKE) visualize_database $(LOG)
	@montage \
		$(BENCHMARK)/$(SAMPLEID)/$(SAMPLEID).png \
		$(WORKDIR)/$(FIGDIR)/$(SAMPLEID)-StableVariance.png \
		-tile 2x1 -geometry +0+0 $(WORKDIR)/$(FIGDIR)/$(SAMPLEID)-benchmark.png
	@echo "=============================================" $(LOG)

build_database: $(LOGFILE) $(PYTHON) $(DATA) $(MAGEMIN)
	@echo "Building MAGEMin database ..." $(LOG)
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
	$(LOG)
	@echo "=============================================" $(LOG)

$(MAGEMIN): $(LOGFILE) $(PYTHON) $(CONFIG)
	@if [ ! -e "$(MAGEMIN)" ]; then \
		echo "=============================================" $(LOG); \
	  echo "Cloning MAGEMin from $(MAGEMIN_REPO) ..." $(LOG); \
	  chmod +x python/clone-magemin.py; \
	  $(CONDA_PYTHON) python/clone-magemin.py $(LOG); \
	  if [ "$(UNAME_S)" = "Linux" ]; then \
	    echo "Configuring MAGEMin for meso ..." $(LOG); \
	    cp $(CONFIG) MAGEMin/Makefile; \
	  fi; \
	  echo "Compiling MAGEMin ..." $(LOG); \
	  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG); \
	  (cd MAGEMin && make) $(LOG); \
	  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" $(LOG); \
	else \
	  echo "MAGEMin found at: $(MAGEMIN)..." $(LOG); \
	fi

remove_conda_env: $(LOGFILE)
	@echo "Removing conda env $(CONDA_ENV_NAME) ..." $(LOG)
	@conda remove --name $(CONDA_ENV_NAME) --all --yes $(LOG)
	@echo "=============================================" $(LOG)

create_conda_env: $(LOGFILE) $(CONDA_SPECS_FILE) find_conda_env
	@if [ "$(HAS_CONDA)" = "false" ]; then \
	  echo "conda not found in PATH, install conda first ..." $(LOG); \
	  echo "See $(GITHUB_REPO) for details ..." $(LOG); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
		echo "Env \"$(CONDA_ENV_NAME)\" found in: $(MY_ENV_DIR) ..." $(LOG); \
	else \
		echo "Creating env $(CONDA_ENV_NAME) from $(CONDA_SPECS_FILE) ..." $(LOG); \
		conda env create --file $(CONDA_SPECS_FILE) > /dev/null $(LOG); \
	  echo "Conda env created ..." $(LOG); \
	fi

find_conda_env:
	@echo "Looking for conda env ..." $(LOG)
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDA_ENV_NAME) | awk '{print $$2}'))

$(PERPLEX): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py $(LOG); \
	else \
	  echo "Perple_x found at: $(PERPLEX) ..." $(LOG); \
	fi

$(DATA): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATA)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py $(LOG); \
	else \
	  echo "Datafiles found at: $(DATA) ..." $(LOG); \
	fi

$(CONFIG): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIG)" ]; then \
		$(CONDA_PYTHON) python/download-assets.py $(LOG); \
		echo "=============================================" $(LOG); \
	else \
	  echo "Meso configuration found at: $(CONFIG) ..." $(LOG); \
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