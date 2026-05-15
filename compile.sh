#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
echo $SCRIPT_DIR
cd $SCRIPT_DIR/src/thesis
mkdir -p .aux/text
latexmk -auxdir=.aux -pdflatex=lualatex -pdf ctufit-thesis.tex
mv ./ctufit-thesis.pdf ../../text/thesis.pdf
# clean artifacts for full compilation
#rm -rf .aux