import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(".."))


def get_version():
    import optimix

    return optimix.__version__


def get_name():
    import optimix

    return optimix.__name__


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]

source_suffix = ".rst"

master_doc = "index"

project = get_name()
copyright = "2018, Danilo Horta"
author = "Danilo Horta"

version = get_version()
release = version

language = None

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py"]

pygments_style = "default"

todo_include_todos = False

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_sidebars = {"**": ["relations.html", "searchbox.html"]}

htmlhelp_basename = "{}doc".format(project)

man_pages = [(master_doc, get_name(), "{} documentation".format(project), [author], 1)]

intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}
