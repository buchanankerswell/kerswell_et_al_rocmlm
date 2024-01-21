# Logging config
DATE = $(shell date +"%d-%m-%Y")
LOGFILE := log/log-$(DATE)
LOG := 2>&1 | tee -a $(LOGFILE)
# Conda config
CONDAENVNAME = rocmlm
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
# Python scripts
PYTHON = \
				 python/build-gfems.py \
				 python/initialize.py \
				 python/gfem.py \
				 python/create-mixing-arrays.py \
				 python/pca.py \
				 python/rocmlm.py \
				 python/scripting.py \
				 python/train-rocmlms.py \
				 python/visualize.py
# Cleanup directories
DATAPURGE = \
						log \
						python/__pycache__ \
						$(DATADIR)/synthetic*.csv \
						$(DATADIR)/gfem-analysis.csv \
						$(DATADIR)/gfem-analysis.csv \
						$(DATADIR)/lut-efficiency.csv \
						$(DATADIR)/earthchem-counts.csv \
						$(DATADIR)/rocmlm-performance.csv \
						$(DATADIR)/earthchem-samples-pca.csv \
						$(DATADIR)/benchmark-samples-pca.csv
DATACLEAN = assets MAGEMin Perple_X gfems rocmlms
FIGSPURGE =
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) initialize
	@$(MAKE) mixing_arrays
	@$(MAKE) gfems
	@$(MAKE) rocmlms

initialize: $(LOGFILE) $(PYTHON) create_conda_env get_assets
	@echo "=============================================" $(LOG)
	@echo "Run the following in order:" $(LOG)
	@echo "    make mixing_arrays" $(LOG)
	@echo "    make gfems" $(LOG)
	@echo "    make rocmlms" $(LOG)
	@echo "=============================================" $(LOG)

rocmlms: initialize
	@PYTHONWARNINGS="ignore" $(CONDAPYTHON) -u python/train-rocmlms.py $(LOG)
	@echo "=============================================" $(LOG)

test: initialize
	@$(CONDAPYTHON) -u python/test.py $(LOG)
	@echo "=============================================" $(LOG)

gfems: initialize
	@$(CONDAPYTHON) -u python/build-gfems.py $(LOG)
	@echo "=============================================" $(LOG)

mixing_arrays: initialize
	@$(CONDAPYTHON) -u python/create-mixing-arrays.py $(LOG)
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
		echo "See: https://github.com/buchanankerswell/kerswell_et_al_rocmlm" $(LOG); \
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

.PHONY: clean purge find_conda_env create_conda_env remove_conda_env get_assets mixing_arrays gfems rocmlms init all