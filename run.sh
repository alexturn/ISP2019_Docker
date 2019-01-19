#!/bin/bash
cd code && python generate_plot.py && python visualize.py;
cd ../latex && pdflatex paper.tex && cp paper.pdf ../results/;