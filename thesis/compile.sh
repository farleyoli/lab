#!/bin/bash

platex report.tex && bibtex report && dvipdfm report.dvi && platex report.tex && dvipdfm report.dvi && bibtex report && platex report.tex && dvipdfm report.dvi && open report.pdf
