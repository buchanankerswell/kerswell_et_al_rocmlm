# System check
ifeq ($(shell uname -s),Linux)
    SYSTEM = linux
else ifeq ($(shell uname -s),Darwin)
    SYSTEM = macos
else
    $(error Unsupported operating system: $(shell uname -s))
endif
# Logging config
DATE = $(shell date +"%d-%m-%Y")
LOGFILE := log/log-$(DATE)
LOG := 2>&1 | tee -a $(LOGFILE)
# Conda config
CONDAENVNAME = rocml
HASCONDA := $(shell command -v conda > /dev/null && echo true || echo false)
CONDASPECSFILE = python/conda-environment.yaml
CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
ifeq ($(shell uname -s),Linux)
    CONDAPYTHON = $$(TMPDIR=$$HOME/scratch/.tmp conda run -n $(CONDAENVNAME) which python)
else ifeq ($(shell uname -s),Darwin)
    CONDAPYTHON = $$(conda run -n $(CONDAENVNAME) which python)
else
    $(error Unsupported operating system: $(shell uname -s))
endif
# Magemin programs
MAGEMIN = MAGEMin
# Perplex programs
PERPLEXDIR = assets/perplex
# Directories with data and scripts
DATADIR = assets/data
CONFIGDIR = assets/config
# Samples
BENCHMARK = $(DATADIR)/benchmark-samples.csv
SYNTHETIC = $(DATADIR)/synthetic-samples-pca3-clusters13.csv
# Python scripts
PYTHON = \
				 python/build-gfem-models.py \
				 python/clone-magemin.py \
				 python/download-assets.py \
				 python/gfem.py \
				 python/create-mixing-arrays.py \
				 python/pca.py \
				 python/rocml.py \
				 python/scripting.py \
				 python/session-info.py \
				 python/submit-jobs.py \
				 python/train-rocml-models.py \
				 python/visualize.py
# Cleanup directories
DATAPURGE = \
						log \
						runs \
						python/__pycache__ \
						$(DATADIR)/benchmark-rocml-performance.csv
DATACLEAN = assets MAGEMin
FIGSPURGE = figs
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) create_conda_env assets $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) -u python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@$(MAKE) buiild_benchmark_datasets
	@$(MAKE) train_benchmark_models

init: $(LOGFILE) $(PYTHON) create_conda_env assets $(MAGEMIN)
	@echo "=============================================" $(LOG)
	@$(CONDAPYTHON) -u python/session-info.py $(LOG)
	@echo "=============================================" $(LOG)
	@echo "Run the following in order:" $(LOG)
	@echo "    make build_benchmark_datasets" $(LOG)
	@echo "    make train_benchmark_models" $(LOG)
	@echo "To clean up the directory:" $(LOG)
	@echo "    make purge" $(LOG)
	@echo "    make remove_conda_env" $(LOG)
	@echo "=============================================" $(LOG)

train_benchmark_models: $(LOGFILE) $(PYTHON) assets
	@$(CONDAPYTHON) -u python/train-rocml-models.py --source '$(BENCHMARK)' --res 64 $(LOG)
	@echo "=============================================" $(LOG)

build_benchmark_datasets: $(LOGFILE) $(PYTHON) assets $(MAGEMIN)
	@$(CONDAPYTHON) -u python/build-gfem-models.py --source '$(BENCHMARK)' --res 64 $(LOG)
	@echo "=============================================" $(LOG)

build_earthchem_datasets: $(LOGFILE) $(PYTHON) assets $(MAGEMIN) create_mixing_arrays
	@$(CONDAPYTHON) -u python/build-gfem-models.py --source '$(SYNTHETIC)' --res 64 $(LOG)
	@echo "=============================================" $(LOG)

create_mixing_arrays:  $(LOGFILE) $(PYTHON) assets
	@$(CONDAPYTHON) -u python/create-mixing-arrays.py --res 64 $(LOG)
	@echo "=============================================" $(LOG)

submit_jobs: $(LOGFILE) $(PYTHON) $(DATADIR)
	@echo "Submitting job to SLURM ..." $(LOG)
	@$(CONDAPYTHON) -u python/submit-jobs.py $(LOG)

$(MAGEMIN): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(MAGEMIN)" ]; then \
		if [ "$(SYSTEM)" == "macos" ]; then \
			$(CONDAPYTHON) -u python/clone-magemin.py --hpc False $(LOG); \
		else \
			$(CONDAPYTHON) -u python/clone-magemin.py --hpc True $(LOG); \
		fi \
	else \
		echo "MAGEMin programs found!" $(LOG); \
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

find_conda_env: $(LOGFILE)
	$(eval MYENVDIR := $(shell conda env list | grep $(CONDAENVNAME) | awk '{print $$2}'))

assets: $(PERPLEXDIR) $(DATADIR) $(CONFIGDIR)

$(PERPLEXDIR): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) -u python/download-assets.py $(LOG); \
	else \
		echo "Perplex programs found!" $(LOG); \
	fi

$(DATADIR): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) -u python/download-assets.py $(LOG); \
	else \
		echo "Data files found!" $(LOG); \
	fi

$(CONFIGDIR): $(LOGFILE) $(PYTHON)
	@if [ ! -e "$(CONFIGDIR)" ]; then \
		$(CONDAPYTHON) -u python/download-assets.py $(LOG); \
	else \
		echo "Config files found!" $(LOG); \
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

.PHONY: purge clean find_conda_env remove_conda_env create_conda_env submit_jobs assets create_mixing_arrays build_earthchem_datasets build_benchmark_datasets train_benchmark_models init all