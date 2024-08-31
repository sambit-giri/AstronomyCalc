# AstronomyCalc

![CI Status](https://github.com/sambit-giri/AstronomyCalc/actions/workflows/ci.yml/badge.svg)
[![GitHub Repository](https://img.shields.io/github/repo-size/sambit-giri/AstronomyCalc)](https://github.com/sambit-giri/AstronomyCalc)
[![PyPI version](https://badge.fury.io/py/AstronomyCalc.svg)](https://badge.fury.io/py/AstronomyCalc)

This package introduces students to basic astronomical calculations and data analysis methods. More detailed documentation is available on the [ReadTheDocs](https://AstronomyCalc.readthedocs.io/) page. Additional Jupyter notebooks, designed to help learners explore concepts by adjusting parameters and observing their effects, will be regularly added to the [notebooks](https://github.com/sambit-giri/AstronomyCalc/tree/main/notebooks) folder.


## Package details

The package provides tools for calculating and solving the following quantities and equations.

* **Friedmann equation**
* **Cosmological distances**
    * Comoving distance
    * Proper distance
    * Light-travel distance 
    * Luminosity distance
    * Angular diameter distance
* **Age of the universe**
* **Cosmological parameter inference**
* **Galaxy rotation curves**

## INSTALLATION

One can install a stable version of this package using pip by running the following command::

    pip install AstronomyCalc

This package is being actively under-development, which involves addition of new modules and bug fixes. In order to use the latest version, one can clone this package.

To install the package from source, one should clone this package running the following::

    git clone https://github.com/sambit-giri/AstronomyCalc.git

To install the package in the standard location, run the following in the root directory::

    python setup.py install

In order to install it in a separate directory::

    python setup.py install --home=directory

One can also install the latest version using pip by running the following command::

    pip install git+https://github.com/sambit-giri/AstronomyCalc.git

The dependencies should be installed automatically during the installation process. The list of required packages can be found in the requirements.txt file present in the root directory.

### Tests

For testing, one can use [pytest](https://docs.pytest.org/en/stable/). To run all the test script, run the either of the following::

    python -m pytest tests
    
## CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/sambit-giri/AstronomyCalc/issues). The issue page is also good if you seek help or have suggestions for us. For more details, please see [here](https://AstronomyCalc.readthedocs.io/contributing.html).
