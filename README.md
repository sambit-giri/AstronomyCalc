# AstronomyCalc

This is a package for basic astronomical and cosmological calculations. More documentation can be found at its [readthedocs](https://AstronomyCalc.readthedocs.io/) page.


## Package details

There are various manipulation and analysis moduled in `Tools21cm`. 

* Angular coordinates: methods to convert data between physical (cMpc) coordinates and observational (angular-frequency) coordinates

The package provides tools for calculating and solving the following quantities and equations.
    
    * Friedmann equation
    * Cosmological distances
        ** Comoving distance
        ** Proper distance
        ** Light-travel distance 
        ** Luminosity distance
        ** Angular diameter distance
    * Age of the universe

## INSTALLATION

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

    
