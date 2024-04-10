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
# Directories with data and perplex configs
DATADIR = assets/data
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
				 python/visualize.py \
				 python/write-md-tables.py
# Cleanup directories
DATAPURGE = \
						log \
						python/__pycache__ \
						$(DATADIR)/synthetic*.csv \
						$(DATADIR)/lut-efficiency.csv \
						$(DATADIR)/earthchem-counts.csv \
						$(DATADIR)/gfem-accuracy-vs-prem.csv \
						$(DATADIR)/earthchem-samples-pca.csv \
						$(DATADIR)/benchmark-samples-pca.csv
DATACLEAN = assets MAGEMin Perple_X gfems rocmlms
FIGSPURGE =
FIGSCLEAN = figs

all: $(LOGFILE) $(PYTHON) gfems rocmlms

write_md_tables: $(LOGFILE) $(PYTHON)
	@$(CONDAPYTHON) -u python/write-md-tables.py $(LOG)
	@echo "=============================================" $(LOG)

rocmlms: $(LOGFILE) $(PYTHON) mixing_arrays
	@PYTHONWARNINGS="ignore" $(CONDAPYTHON) -u python/train-rocmlms.py $(LOG)
	@echo "=============================================" $(LOG)

gfems: mixing_arrays
	@$(CONDAPYTHON) -u python/build-gfems.py $(LOG)
	@echo "=============================================" $(LOG)

mixing_arrays: initialize
	@if [ ! -e "$(DATADIR)/benchmark-samples-pca.csv" ]; then \
		$(CONDAPYTHON) -u python/create-mixing-arrays.py $(LOG); \
	else \
		echo "Mixing arrays found!" $(LOG); \
	fi
	@echo "=============================================" $(LOG)

initialize: $(LOGFILE) $(PYTHON) create_conda_env get_assets

get_assets: $(DATADIR)

$(DATADIR): $(LOGFILE) $(PYTHON)
	@if [ ! -d "$(DATADIR)" ]; then \
		$(CONDAPYTHON) -u python/initialize.py $(LOG); \
	else \
		echo "GFEM programs and data files found!" $(LOG); \
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

.PHONY: clean purge find_conda_env create_conda_env remove_conda_env get_assets mixing_arrays gfems rocmlms initialize write_md_tables all