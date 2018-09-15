# BartPy

[![Build Status](https://travis-ci.org/JakeColtman/bartpy.svg?branch=master)](https://travis-ci.org/JakeColtman/bartpy)

### Introduction

BartPy is a pure python implementation of the Bayesian additive regressions trees model of Chipman et al [1].

### Reasons to use BART

 * Much less parameter optimization required that GBT
 * Provides confidence intervals in addition to point estimates
 * Extremely flexible through use of priors and embedding in bigger models

### Reasons to use the library:

 * Can be plugged into existing sklearn pipelines
 * Designed to be extremely easy to modify and extend

### Trade offs:

 * Speed - BartPy is significantly slower than other BART libraries
 * Memory - BartPy uses a lot of caching compared to other approaches
 * Instability - the library is still under construction

### Alternative libraries

 * R - https://cran.r-project.org/web/packages/bartMachine/bartMachine.pdf
 * R - https://cran.r-project.org/web/packages/BayesTree/index.html

### References

 [1] https://arxiv.org/abs/0806.3286
 [2] http://www.gatsby.ucl.ac.uk/~balaji/pgbart_aistats15.pdf
 [3] https://arxiv.org/ftp/arxiv/papers/1309/1309.1906.pdf
 [4] https://cran.r-project.org/web/packages/BART/vignettes/computing.pdf
