# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os, sys
sys.path.insert(0,os.path.abspath('../..'))

project = 'resolvent4py'
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

html_logo = "logo.png"
html_show_sphinx = False

templates_path = ['_templates']
exclude_patterns = []

rst_prolog = """
.. |pkgname| replace:: resolvent4py
.. _MatType: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Mat.Type.html
.. _MatSizeSpec: https://petsc.org/release/petsc4py/reference/petsc4py.typing.MatSizeSpec.html#petsc4py.typing.MatSizeSpec
.. _KSPType: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.Type.html#petsc4py.PETSc.KSP.Type
.. _KSP: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.KSP.html
.. _Vec: https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Vec.html
.. _LayoutSizeSpec: https://petsc.org/release/petsc4py/reference/petsc4py.typing.LayoutSizeSpec.html#petsc4py.typing.LayoutSizeSpec
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add custom CSS
# def setup(app):
#     app.add_css_file('custom.css')