# Build

1. pdflatex -interaction nonstopmode -file-line-error main.tex
2. biber main
3. pdflatex -interaction nonstopmode -file-line-error main.tex
4. pdflatex -interaction nonstopmode -file-line-error main.tex
