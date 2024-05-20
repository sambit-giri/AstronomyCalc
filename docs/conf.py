# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
import sphinx_readable_theme

# Add the path to the source code directory to sys.path
sys.path.insert(0, os.path.abspath('../src/'))

# -- Project information -----------------------------------------------------

project = 'AstronomyCalc'
copyright = '2022, Sambit Giri'
author = 'Sambit Giri'
version = release = '2.1'

# -- General configuration ---------------------------------------------------

# Minimal Sphinx version required
# needs_sphinx = '2.3'

# Sphinx extension module names
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel',
    'numpydoc',
    'nbsphinx',
]

# Paths that contain templates
templates_path = ['templates']

# Patterns to exclude when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'templates']

# -- Options for HTML output -------------------------------------------------

# Theme configuration
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_theme = 'readable'  # Changed to 'readable' since 'alabaster' was commented out
pygments_style = 'trac'

# HTML options
html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
html_sidebars = {
    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html']
}
html_short_title = f'{project}'

# Napoleon options
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

# MathJax configuration
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
