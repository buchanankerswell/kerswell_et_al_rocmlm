MS = kerswell-et-al-rocmlm
SUPP = kerswell-et-al-rocmlm-supporting
DIFF = g3_submission/kerswell-et-al-rocmlm

all: paper supp

paper: $(MS).pdf
	@open $<

supp: $(SUPP).pdf
	@open $<

diff: diff.tex

$(MS).pdf: $(MS).Rmd
	@R -q -e 'rmarkdown::render("$(MS).Rmd", quiet=T, clean=F)'

$(SUPP).pdf: $(SUPP).Rmd
	@R -q -e 'rmarkdown::render("$(SUPP).Rmd", quiet=T, clean=F)'

diff.tex: $(MS).tex $(DIFF).tex
	@printf "\nlatexdiff for %s vs. %s ..." "$(DIFF).tex" "$(MS).tex"
	@latexdiff --type=CFONT --disable-citation-markup --graphics-markup=none --allow-spaces $(DIFF).tex $(MS).tex > diff.tex

get_tables:
	@$(MAKE) -C .. write_md_tables

purge:
	@rm -f *.{html,acr,alg,gz,glg,gls,ilg,nls,acn,glo,ist,lof,lot,nlo,aux,dvi,log,bbl,blg,brf,fls,toc,thm,out,fdb_latexmk,pdf}

clean: purge

.PHONY: clean purge get_tables diff supp paper all