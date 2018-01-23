#!/bin/bash

platex report.tex && bibtex report && dvipdfm report.dvi && makeindex report.nlo -s nomencl.ist -o report.nls && platex report.tex && dvipdfm report.dvi && bibtex report && platex report.tex && dvipdfm report.dvi && open report.pdf
