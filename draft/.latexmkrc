# Custom dependency for glossaries package
add_cus_dep( 'glo', 'gls', 0, 'makeglo2gls' );
sub makeglo2gls {
    system( "makeindex -s \"$_[0].ist\" -t \"$_[0].glg\" -o \"$_[0].gls\" \"$_[0].glo\" > /dev/null 2>&1" );
}

# Custom dependency for glossaries[acronym] package
add_cus_dep( 'acn', 'acr', 0, 'makeacn2acr' );
sub makeacn2acr {
    system( "makeindex -s \"$_[0].ist\" -t \"$_[0].alg\" -o \"$_[0].acr\" \"$_[0].acn\" > /dev/null 2>&1" );
}

# Custom dependency and function for nomencl package
add_cus_dep( 'nlo', 'nls', 0, 'makenlo2nls' );
sub makenlo2nls {
    system( "makeindex -s nomencl.ist -o \"$_[0].nls\" \"$_[0].nlo\" > /dev/null 2>&1" );
}

# Maximum number of runs before bailing
$max_repeat = 10;

# pdflatex call
$pdflatex = 'pdflatex --shell-escape -interaction=nonstopmode -file-line-error -synctex=1 %O %S > /dev/null 2>&1';

# Clean additional files
$clean_ext = "bbl acn acr alg glg glo gls ist nlo nls";