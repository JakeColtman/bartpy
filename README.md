# BartPy

[![Build Status](https://travis-ci.org/JakeColtman/bartpy.svg?branch=master)](https://travis-ci.org/JakeColtman/bartpy)

### Introduction

BartPy is a pure python implementation of the Bayesian additive regressions trees model of Chipman et al [1].

### Reasons to use BART

 * Much less parameter optimization required that GBT
 * Provides confidence intervals in addition to point estimates
 * Extremely flexible through use of priors and embedding in bigger models

### Reasons to use the library:

 * Can be plugged into existing sklearn workflows
 * Everything is done in pure python, allowing for easy inspection of model runs
 * Designed to be extremely easy to modify and extend

### Trade offs:

 * Speed - BartPy is significantly slower than other BART libraries
 * Memory - BartPy uses a lot of caching compared to other approaches
 * Instability - the library is still under construction

### How to use:

There are two main APIs for BaryPy:
  1. High level sklearn API
  2. Low level access for implementing custom conditions

If possible, it is recommended to use the sklearn API until you reach something that can't be implemented that way.  The API is easier, shared with other models in the ecosystem, and allows simpler porting to other models.

#### Sklearn API

The high level API works as you would expect

``` python
from bartpy.sklearnmodel import SklearnModel
model = SklearnModel() # Use default parameters
model.fit(X, y) # Fit the model
predictions = model.predict() # Make predictions on the train set
out_of_sample_predictions = model.predict(X_test) # Make predictions on new data
```

The model object can be used in all of the standard sklearn tools, e.g. cross validation and grid search

```python
from bartpy.sklearnmodel import SklearnModel
model = SklearnModel() # Use default parameters
cross_validate(model)
```

##### Extensions

BartPy offers a number of convenience extensions to base BART.  The most prominent of these is using BART to predict the residuals of a base model.
It is most natural to use a linear model as the base, but any sklearn compatible model can be used

```python
from bartpy.extensions.baseestimator import ResidualBART
model = ResidualBART(base_estimator=LinearModel())
model.fit(X, y)
```

A nice feature of this is that we can combine the interpretability of a linear model with the power of a trees model

#### Lower level API

BartPy is designed to expose all of its internals, so that it can be extended and modifier.  In particular, using the lower level API it is possible to:
  * Customize the set of possible tree operations (prune and grow by default)
  * Control the order of sampling steps within a single Gibbs update
  * Extend the model to include additional sampling steps

Some care is recommended when working with these type of changes.  Through time the process of changing them will become easier, but today they are somewhat complex

If all you want to customize are things like priors and number of trees, it is much easier to use the sklearn API

### Alternative libraries

 * R - https://cran.r-project.org/web/packages/bartMachine/bartMachine.pdf
 * R - https://cran.r-project.org/web/packages/BayesTree/index.html

### References

 [1] https://arxiv.org/abs/0806.3286
 [2] http://www.gatsby.ucl.ac.uk/~balaji/pgbart_aistats15.pdf
 [3] https://arxiv.org/ftp/arxiv/papers/1309/1309.1906.pdf
 [4] https://cran.r-project.org/web/packages/BART/vignettes/computing.pdf
