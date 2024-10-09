# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os, sys
sys.path.insert(0,os.path.abspath('../..'))

project = 'LinToolbox4py'
copyright = '2024, Alberto Padovan'
author = 'Alberto Padovan'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

rst_prolog = """
.. _matSizeSpec: https://petsc.org/release/petsc4py/reference/petsc4py.typing.MatSizeSpec.html#petsc4py.typing.MatSizeSpec
"""



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add custom CSS
# def setup(app):
#     app.add_css_file('custom.css')