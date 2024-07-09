MS = kerswell-et-al-rocmlm
SUPP = kerswell-et-al-rocmlm-supporting

all: paper supp

paper: $(MS).pdf
	@open $<

supp: $(SUPP).pdf
	@open $<

$(MS).pdf: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(SUPP).pdf: $(SUPP).Rmd
	@R -q -e 'rmarkdown::render("$(SUPP).Rmd", quiet=T, clean=F)'

get_tables:
	@$(MAKE) -C .. write_md_tables

purge:
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk,pdf} *.knit.md $(MS).tex $(SUPP).tex

clean: purge

.PHONY: clean purge get_tables supp paper all