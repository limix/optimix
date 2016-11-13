# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import sphinx_rtd_theme

try:
    import optimix
    version = optimix.__version__
except ImportError:
    version = 'unknown'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = True
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'optimix'
copyright = '2016, Danilo Horta'
author = 'Danilo Horta'
release = version
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'default'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = 'optimixdoc'
latex_elements = {}
latex_documents = [
    (master_doc, 'optimix.tex', 'optimix Documentation',
     'Danilo Horta', 'manual'),
]
man_pages = [
    (master_doc, 'optimix', 'optimix Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'optimix', 'optimix Documentation',
     author, 'optimix', 'One line description of project.',
     'Miscellaneous'),
]
intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}
