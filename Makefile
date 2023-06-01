MAGEMIN = MAGEMin/MAGEMin
PYTHON = python/mad-create.py python/functions.py
DATAPURGE = MAGEMin/*.dat output
DATACLEAN = log
FIGSPURGE = figs

all: $(MAGEMIN) $(PYTHON)
	@./run.sh

$(MAGEMIN): MAGEMin/Makefile
	@echo "Compiling MAGEMin ..."
	@(cd MAGEMin && make) > /dev/null 2>&1

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN)

.PHONY: all purge clean purge