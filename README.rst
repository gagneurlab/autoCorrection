========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - | |Version| |Build| |Wheel| |License|


.. |version| image:: https://img.shields.io/pypi/v/autoCorrection.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/autoCorrection


.. |Build| image::  https://travis-ci.org/gagneurlab/autoCorrection.svg?branch=master
    :alt: Build status
    :target: https://travis-ci.org/gagneurlab/autoCorrection


.. |wheel| image:: https://img.shields.io/pypi/wheel/autoCorrection.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/autoCorrection

.. |License| image:: https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000
    :alt: License MIT
    :target: https://github.com/gagneurlab/autoCorrection/blob/master/LICENSE


.. end-badges



* Free software: MIT license

Activate virtual environment
==================
Together with the autoCorrection package you will get

        'tensorflow',
        'keras',
        'numpy',
        'kopt',
        'scipy',
        'h5py',
        'sklearn',
        'pandas',
        'statsmodels',
        'pytest'

packages automatically installed, if not present.

If you don't wannt to install these packages globally, please use virtual environment.

If you have problems with virtualenv, installing using conda may help:

(Installation of conda: https://conda.io/docs/user-guide/install/index.html)

Make sure you are using python 3.

    conda create -n mypyth3 python=3.6

    source activate mypyth3

    conda install virtualenv

activate new environment in active python 3 environment:

    virtualenv env-with-autoCorrection

    source env-with-autoCorrection/bin/activate

Check if you are still using python 3:

    python --version


Package Installation
============

::

    pip install autoCorrection


Deactivate virtual environment
============

::

    deactivate

Usage
============

::

    #in python:
    python
    import autoCorrection
    import numpy as np
    counts = np.random.negative_binomial(n = 20, p=0.2, size = (10,8))
    sf = np.ones((10,8))
    corrector = autoCorrection.correctors.AECorrector()
    c = corrector.correct(counts = counts, size_factors = sf)

    #in R:
    library(reticulate)
    autoCorrection <- import("autoCorrection")
    corrected <- autoCorrection$correctors$AECorrector(model_name, model_directory)$correct(COUNTS, SIZE_FACTORS, only_predict=FALSE)

Documentation
=============

https://i12g-gagneurweb.in.tum.de/public/docs/autocorrection/


