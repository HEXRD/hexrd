# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

from hexrd.core.constants import __version__ as version

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'hexrd'
copyright = '2023, LLNL, AFRL, CHESS, and Kitware'
author = 'Joel Bernier, Patrick Avery, Saransh Singh, Chris Harris, Rachel Lim, Darren Pagan, Donald Boyce, Brianna Major, John Tourtellot, Óscar Villellas Guillén, Ryan Rygg, Kelly Nygren, Paul Shade'

# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # For automatic documentation of modules
    'sphinx.ext.autodoc',
    # The nicer looking readthedocs theme
    'sphinx_rtd_theme',
    # Support for markdown files
    'myst_parser',
    # Add a link to the source code next to every documentation entry
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
