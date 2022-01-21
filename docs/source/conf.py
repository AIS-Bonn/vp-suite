# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

def get_project_info():
    from vp_suite.__about__ import __version__ as release,\
        __name__ as project, \
        __author__ as author, \
        __copyright_short__ as copyright
    return project, copyright, author, release

project, copyright, author, release = get_project_info()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'm2r2',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

add_module_names = False
autodoc_preserve_defaults = True
napoleon_use_param = True

autodoc_default_flags = ['members', 'show-inheritance']
autodoc_default_options = {
    'special-members': '__init__',
}

intersphinx_mapping = {
    'python':   ('https://docs.python.org/3', None),
    'PIL':      ('https://pillow.readthedocs.io/en/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['vp_suite/resources/*']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = ['.rst', '.md']
