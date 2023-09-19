# Makefile config
SHELL = /bin/bash -o pipefail
UNAMES := $(shell uname -s)
DATE = $(shell date +"%d-%m-%Y")
DATELONG = $(shell date +"%d-%B-%Y")
# Logging
LOGFILE := log/log-$(shell date +"%d-%m-%Y")
LOG := 2>&1 | tee -a $(LOGFILE)
# Conda env
CONDAENVNAME = rocml
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
# Magemin program
MAGEMIN = MAGEMin
# Perplex program
PERPLEX = assets/perplex
# Directories with data and scripts
DATADIR = assets/data
CONFIG = assets/config
PYTHON = python/build-magemin-dataset.py \
				 python/earthchem-samples-pca.py \
				 python/clone-magemin.py \
				 python/download-assets.py \
				 python/rocml.py \
				 python/train-benchmark-rocmls.py \
				 python/session-info.py \
				 python/submit-jobs.py \
				 python/visualize-dataset.py \
				 python/visualize-other.py \
# Dataset build options
SOURCE ?= $(DATADIR)/benchmark-samples.csv
SAMPLEID ?= PUM
PMIN ?= 1
PMAX ?= 28
TMIN ?= 773
TMAX ?= 2273
RES ?= 8
EMSONLY ?= True
DATASET ?= train
NORMOX ?= all
SEED = 42
PARALLEL ?= True
NPROCS ?= $(shell expr $(shell nproc) - 2)
KFOLDS ?= $(shell expr $(shell nproc) - 2)
OUTDIR ?= runs
VERBOSE ?= False
# RocML options
MLPARAMS ?= ["DensityOfFullAssemblage", "Vp", "Vs", "LiquidFraction"]
MLMODS ?= ["KN", "RF", "DT", "NN1", "NN2", "NN3"]
MLTUNE ?= False
# Bulk rock composition sampling options
OXIDES ?= ["SIO2", "AL2O3", "CAO", "MGO", "FEO", "K2O", "NA2O", "TIO2", "FE2O3", "CR2O3"]
NPCA ?= 3
KCLUSTER ?= 3
# Visualization options
FIGDIR ?= figs
PARAMS ?= ["StableSolutions", "StableVariance", "DensityOfFullAssemblage"]
COLORMAP ?= bone
# Cleanup directories
DATAPURGE = python/__pycache__ \
						.job \
						output \
						$(DATADIR)/*assemblages.csv \
						$(DATADIR)/benchmark-rocmls-performance.csv \
						$(DATADIR)/benchmark-gfem-efficiency-$(DATE).csv
DATACLEAN = assets log MAGEMin runs
FIGSPURGE =
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env $(DATADIR) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@$(MAKE) earthchem_samples_pca
	@$(MAKE) magemin_dataset DATASET=train
	@$(MAKE) magemin_dataset DATASET=valid
	@$(MAKE) perplex_dataset DATASET=train
	@$(MAKE) perplex_dataset DATASET=valid
	@$(MAKE) train_benchmark_rocmls
	@$(MAKE) visualize_dataset
	@$(MAKE) visualize_other

init: $(LOGFILE) $(PYTHON) create_conda_env $(DATADIR) $(CONFIG) $(PERPLEX) $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@echo "Run the following in order:" $(LOG)
	@echo "    make earthchem_samples_pca" $(LOG)
	@echo "    make magemin_dataset DATASET=train" $(LOG)
	@echo "    make magemin_dataset DATASET=valid" $(LOG)
	@echo "    make perplex_dataset DATASET=train" $(LOG)
	@echo "    make perplex_dataset DATASET=valid" $(LOG)
	@echo "    make train_benchmark_rocmls" $(LOG)
	@echo "    make visualize_dataset" $(LOG)
	@echo "    make visualize_other" $(LOG)
	@echo "To clean up the directory:" $(LOG)
	@echo "    make purge" $(LOG)
	@echo "    make remove_conda_env" $(LOG)
	@echo "=============================================" $(LOG)

visualize_other: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/visualize-other.py \
		--sampleid '$(SAMPLEID)' \
		--params '$(MLPARAMS)' \
		--res $(RES) \
		$(LOG)
	@echo "=============================================" $(LOG)

visualize_dataset: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/visualize-dataset.py \
		--sampleid '$(SAMPLEID)' \
		--res $(RES) \
		--dataset $(DATASET) \
		--params '$(PARAMS)' \
		--colormap $(COLORMAP) \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
		--datadir $(DATADIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

train_all_benchmark_rocmls: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(MAGEMIN) $(PERPLEX)
	@for sample in DMM NMORB PUM RE46; do \
		for res in 8 16 32 64 128; do \
			$(MAKE) train_benchmark_rocmls SAMPLEID=$$sample RES=$$res; \
		done; \
		$(MAKE) visualize_dataset SAMPLEID=$$sample RES=$$res; \
  done

train_benchmark_rocmls:  $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/train-benchmark-rocmls.py \
		--sampleid '$(SAMPLEID)' \
		--res $(RES) \
		--params '$(MLPARAMS)' \
		--models '$(MLMODS)' \
		--tune $(MLTUNE) \
		--seed $(SEED) \
		--kfolds $(KFOLDS) \
		--parallel $(PARALLEL) \
		--nprocs $(NPROCS) \
		--colormap $(COLORMAP) \
		--outdir $(OUTDIR) \
		--figdir $(FIGDIR) \
		--datadir $(DATADIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

build_all_benchmarks: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(MAGEMIN) $(PERPLEX)
	@for sample in DMM NMORB PUM RE46; do \
		for res in 8 16 32 64 128; do \
			for dataset in train valid; do \
				$(MAKE) magemin_dataset SAMPLEID=$$sample RES=$$res DATASET=$$dataset; \
				$(MAKE) perplex_dataset SAMPLEID=$$sample RES=$$res DATASET=$$dataset; \
			done; \
		done; \
  done

magemin_dataset: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(MAGEMIN)
	@$(CONDAPYTHON) python/build-magemin-dataset.py \
		--Pmin $(PMIN) \
		--Pmax $(PMAX) \
		--Tmin $(TMIN) \
		--Tmax $(TMAX) \
		--res $(RES) \
		--source $(SOURCE) \
		--sampleid $(SAMPLEID) \
		--normox '$(NORMOX)' \
		--dataset $(DATASET) \
		--emsonly $(EMSONLY) \
		--parallel $(PARALLEL) \
		--nprocs $(NPROCS) \
		--outdir $(OUTDIR) \
		--datadir $(DATADIR) \
		--logfile log/log-magemin-$(SAMPLEID)-$(DATASET)-$(RES)-$(DATE) \
		--verbose $(VERBOSE) \
		$(LOG)
	@echo "=============================================" $(LOG)
	@echo "To visualize dataset, run:" $(LOG)
	@echo "make visualize_dataset SAMPLEID=$(SAMPLEID) RES=$(RES)" $(LOG)
	@echo "=============================================" $(LOG)

perplex_dataset: $(LOGFILE) $(PYTHON) $(DATADIR) $(CONFIG) $(PERPLEX)
	@$(CONDAPYTHON) python/build-perplex-dataset.py \
		--Pmin $(PMIN) \
		--Pmax $(PMAX) \
		--Tmin $(TMIN) \
		--Tmax $(TMAX) \
		--res $(RES) \
		--source $(SOURCE) \
		--sampleid $(SAMPLEID) \
		--normox '$(NORMOX)' \
		--dataset $(DATASET) \
		--emsonly $(EMSONLY) \
		--parallel $(PARALLEL) \
		--nprocs $(NPROCS) \
		--outdir $(OUTDIR) \
		--datadir $(DATADIR) \
		--configdir $(CONFIG) \
		--perplexdir $(PERPLEX) \
		--logfile log/log-perplex-$(SAMPLEID)-$(DATASET)-$(RES)-$(DATE) \
		--verbose $(VERBOSE) \
		$(LOG)
	@echo "=============================================" $(LOG)
	@echo "To visualize dataset, run:" $(LOG)
	@echo "make visualize_dataset SAMPLEID=$(SAMPLEID) RES=$(RES)" $(LOG)
	@echo "=============================================" $(LOG)

earthchem_samples_pca:  $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) python/earthchem-samples-pca.py \
		--res $(RES) \
		--oxides '$(OXIDES)' \
		--npca $(NPCA) \
		--kcluster $(KCLUSTER) \
		--seed $(SEED) \
		--figdir $(FIGDIR) \
		--datadir $(DATADIR) \
	$(LOG)
	@echo "=============================================" $(LOG)

write_tables: $(LOGFILE)
	@echo "Writing markdown tables ..." $(LOG)
	@$(CONDAPYTHON) python/write-markdown-tables.py

submit_jobs: $(LOGFILE) $(PYTHON) $(DATADIR)
	@echo "Submitting job to SLURM ..." $(LOG)
	@$(CONDAPYTHON) python/submit-jobs.py $(LOG)
	@echo "=============================================" $(LOG)

$(MAGEMIN): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(MAGEMIN)" ]; then \
		echo "=============================================" $(LOG); \
		chmod +x python/clone-magemin.py; \
		$(CONDAPYTHON) python/clone-magemin.py $(LOG); \
	else \
		echo "MAGEMin found at: $(MAGEMIN)" $(LOG); \
		echo "=============================================" $(LOG); \
	fi

remove_conda_env: $(LOGFILE)
	@echo "Removing conda env $(CONDAENVNAME) ..." $(LOG)
	@conda remove --name $(CONDAENVNAME) --all --yes $(LOG)
	@echo "=============================================" $(LOG)

create_conda_env: $(LOGFILE) $(CONDASPECSFILE) find_conda_env
	@if [ "$(HASCONDA)" = "false" ]; then \
		echo "conda not found in PATH, install conda first ..." $(LOG); \
		echo "See https://github.com/buchanankerswell/kerswell_et_al_rocml ..." $(LOG); \
		exit 1; \
	fi
	@if [ -d "$(MY_ENV_DIR)" ]; then \
		echo "Conda environment \"$(CONDAENVNAME)\" found at: $(MY_ENV_DIR)" $(LOG); \
		echo "=============================================" $(LOG); \
	else \
		echo "Creating environment $(CONDAENVNAME) from:" $(LOG); \
		echo "	$(CONDASPECSFILE)" $(LOG); \
		conda env create --file $(CONDASPECSFILE) $(LOG); \
		echo "	Conda environment $(CONDAENVNAME) created ..." $(LOG); \
	fi
	@echo "=============================================" $(LOG)

find_conda_env: $(LOGFILE)
	@echo "Looking for conda environment $(CONDAENVNAME)..." $(LOG)
	$(eval MY_ENV_DIR := $(shell conda env list | grep $(CONDAENVNAME) | awk '{print $$2}'))
	@echo "=============================================" $(LOG)

$(PERPLEX): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Perplex found at: $(PERPLEX)" $(LOG); \
		echo "=============================================" $(LOG); \
	fi

$(DATADIR): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
	else \
		echo "Data files found at: $(DATADIR)" $(LOG); \
		echo "=============================================" $(LOG); \
	fi

$(CONFIG): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIG)" ]; then \
		$(CONDAPYTHON) python/download-assets.py $(LOG); \
		echo "=============================================" $(LOG); \
	else \
		echo "Configuration files found at: $(CONFIG)" $(LOG); \
		echo "=============================================" $(LOG); \
	fi

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir -p log; \
		touch $(LOGFILE); \
	fi

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)

.PHONY: purge clean find_conda_env remove_conda_env create_conda_env submit_jobs write_tables earthchem_samples_pca perplex_dataset magemin_dataset build_all_benchmarks train_benchmark_rocmls train_all_benchmark_rocmls visualize_dataset visualize_other init all