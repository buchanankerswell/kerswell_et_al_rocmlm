# Logging config
DATE = $(shell date +"%d-%m-%Y")
LOGFILE := log/log-$(DATE)
LOG := 2>&1 | tee -a $(LOGFILE)
# Conda config
CONDAENVNAME = rocml
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
# Magemin programs
MAGEMIN = MAGEMin
# Perplex programs
PERPLEX = Perple_X
# Directories with data and scripts
DATADIR = assets/data
CONFIGDIR = assets/config
# GFEM options
BENCHMARK = $(DATADIR)/benchmark-samples.csv
SYNTHETIC = $(DATADIR)/synthetic-samples-pca2-clusters23.csv
RES ?= 32
BATCH ?= 0
NBATCHES ?= 4
NPROCS ?= 8
VIS ?= True
DEBUG ?= True
# Python scripts
PYTHON = \
				 python/build-gfem-models.py \
				 python/initialize.py \
				 python/gfem.py \
				 python/create-mixing-arrays.py \
				 python/pca.py \
				 python/rocml.py \
				 python/scripting.py \
				 python/submit-jobs.py \
				 python/train-rocml-models.py \
				 python/visualize.py
# Cleanup directories
DATAPURGE = \
						log \
						runs \
						python/__pycache__ \
						$(DATADIR)/benchmark-rocml-performance.csv
DATACLEAN = assets MAGEMin Perple_X
FIGSPURGE = figs
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env get_assets
	@echo "=============================================" $(LOG)
	@$(MAKE) benchmark_datasets
	@$(MAKE) benchmark_models

initialize: $(LOGFILE) $(PYTHON) create_conda_env get_assets
	@echo "=============================================" $(LOG)
	@echo "Run the following in order:" $(LOG)
	@echo "    make build_benchmark_datasets" $(LOG)
	@echo "    make train_benchmark_models" $(LOG)
	@echo "To clean up the directory:" $(LOG)
	@echo "    make purge" $(LOG)
	@echo "    make remove_conda_env" $(LOG)
	@echo "=============================================" $(LOG)

benchmark_models: $(LOGFILE) $(PYTHON) get_assets
	@$(CONDAPYTHON) -u python/train-rocml-models.py --source '$(BENCHMARK)' --res $(RES) \
		--visualize $(VIS) $(LOG)
	@echo "=============================================" $(LOG)

earthchem_models: $(LOGFILE) $(PYTHON) get_assets
	@$(CONDAPYTHON) -u python/train-rocml-models.py --source '$(SYNTHETIC)' --res $(RES) \
		--visualize $(VIS) $(LOG)
	@echo "=============================================" $(LOG)

benchmark_datasets: $(LOGFILE) $(PYTHON) get_assets
	@$(CONDAPYTHON) -u python/build-gfem-models.py --source '$(BENCHMARK)' --res $(RES) \
		--batch $(BATCH) --debug $(DEBUG) --visualize $(VIS) $(LOG)
	@echo "=============================================" $(LOG)

earthchem_datasets: $(LOGFILE) $(PYTHON) get_assets
	@for k in 0 1 2 3; do \
		$(CONDAPYTHON) -u python/build-gfem-models.py --source '$(SYNTHETIC)' --res $(RES) \
		--nbatches $(NBATCHES) --batch $$k --nprocs $(NPROCS) --debug $(DEBUG) --visualize $(VIS) \
		$(LOG); \
	done;
	@echo "=============================================" $(LOG)

earthchem_batch: $(LOGFILE) $(PYTHON) get_assets
	@$(CONDAPYTHON) -u python/build-gfem-models.py --source '$(SYNTHETIC)' --res $(RES) \
		--nbatches $(NBATCHES) --batch $(BATCH) --nprocs $(NPROCS) --debug $(DEBUG) --visualize \
		$(VIS) $(LOG)
	@echo "=============================================" $(LOG)

mixing_arrays:  $(LOGFILE) $(PYTHON) get_assets
	@$(CONDAPYTHON) -u python/create-mixing-arrays.py --res $(RES) --visualize $(VIS) $(LOG)
	@echo "=============================================" $(LOG)

get_assets: $(DATADIR) $(CONFIGDIR) $(MAGEMIN) $(PERPLEX)

$(PERPLEX): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) -u python/initialize.py $(LOG); \
	else \
		echo "Perplex programs found!" $(LOG); \
	fi

$(MAGEMIN): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(MAGEMIN)" ]; then \
		$(CONDAPYTHON) -u python/initialize.py $(LOG); \
	else \
		echo "MAGEMin programs found!" $(LOG); \
	fi

$(CONFIGDIR): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIGDIR)" ]; then \
		$(CONDAPYTHON) -u python/initialize.py $(LOG); \
	else \
		echo "Config files found!" $(LOG); \
	fi

$(DATADIR): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) -u python/initialize.py $(LOG); \
	else \
		echo "Data files found!" $(LOG); \
	fi

$(LOGFILE):
	@if [ ! -e "$(LOGFILE)" ]; then \
		mkdir -p log; \
		touch $(LOGFILE); \
	fi

remove_conda_env:
	@echo "Removing conda env $(CONDAENVNAME) ..."
	@conda remove --name $(CONDAENVNAME) --all --yes

create_conda_env: $(LOGFILE) $(CONDASPECSFILE) find_conda_env
	@if [ "$(HASCONDA)" = "false" ]; then \
		echo "Install conda first!" $(LOG); \
		echo "See: https://github.com/buchanankerswell/kerswell_et_al_rocml" $(LOG); \
		exit 1; \
	fi
	@if [ -d "$(MYENVDIR)" ]; then \
		echo "Conda environment \"$(CONDAENVNAME)\" found!" $(LOG); \
	else \
		echo "Creating conda environment $(CONDAENVNAME) ..." $(LOG); \
		conda env create --file $(CONDASPECSFILE) $(LOG) > /dev/null 2>&1; \
		echo "Conda environment $(CONDAENVNAME) created!" $(LOG); \
	fi
	@echo "=============================================" $(LOG)

find_conda_env: $(LOGFILE)
	$(eval MYENVDIR := $(shell conda env list | grep $(CONDAENVNAME) | awk '{print $$2}'))

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(FIGSCLEAN)

.PHONY: purge clean find_conda_env remove_conda_env create_conda_env get_assets mixing_arrays earthchem_batch earthchem_datasets benchmark_datasets earthchem_models benchmark_models init all