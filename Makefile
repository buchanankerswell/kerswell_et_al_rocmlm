MAGEMIN = MAGEMin/MAGEMin
MAGEMIN_MAKE = MAGEMin/Makefile
PYTHON = python/create-mad.py python/functions.py
LOGFILE := log/log-$(shell date +%Y-%m-%d)
DATAPURGE =
DATACLEAN = runs log
FIGSPURGE = figs

all: $(MAGEMIN) $(PYTHON) $(LOGFILE)
	@python/create-mad.py 2>&1 | tee -a $(LOGFILE)
	@python/visualize-mad.py 2>&1 | tee -a $(LOGFILE)

$(MAGEMIN): $(MAGEMIN_MAKE) $(LOGFILE)
	@echo "Compiling MAGEMin ..." 2>&1 | tee -a $(LOGFILE)
	@echo "===========================================================" 2>&1 | tee -a $(LOGFILE)
	@(cd MAGEMin && make) 2>&1 | tee -a $(LOGFILE)
	@echo "===========================================================" 2>&1 | tee -a $(LOGFILE)

$(LOGFILE):
	@mkdir log
	@touch $(LOGFILE)

purge:
	@rm -rf $(DATAPURGE) $(FIGSPURGE)

clean: purge
	@rm -rf $(DATACLEAN) $(MAGEMIN)

.PHONY: all purge clean purge