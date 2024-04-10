MS = kerswell-et-al-rocmlm
SUPP = kerswell-et-al-rocmlm-supporting
DIFF = g3_submission/kerswell-et-al-rocmlm

all: get_tables paper supp

paper: $(MS).Rmd get_tables
	@echo "rendering $(MS) ..."
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)' > /dev/null 2>&1
	@echo "============================================="

supp: $(SUPP).Rmd get_tables
	@echo "rendering $(SUPP) ..."
	@R -q -e 'rmarkdown::render("$(SUPP).Rmd", quiet=T, clean=F)' > /dev/null 2>&1
	@echo "============================================="

diff: $(MS).tex $(DIFF).tex
	@echo "latexdiff for $(DIFF).tex vs. $(MS).tex ..."
	@latexdiff --type=CFONT --disable-citation-markup --graphics-markup=none --allow-spaces \
		$(DIFF).tex $(MS).tex > diff.tex
	@echo "============================================="

get_tables:
	@$(MAKE) -C .. write_md_tables

purge:
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk,tex,md,pdf,md}

clean: purge

.PHONY: clean purge get_tables diff supp paper all