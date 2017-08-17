from __future__ import unicode_literals

import re
from time import strftime

import sphinx_rtd_theme

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser


def get_metadata():
    config = ConfigParser()
    config.read('setup.cfg')
    return dict(config.items('metadata'))


def get_version(metadata):
    expr = re.compile(r"__version__ *= *\"(.*)\"")
    prjname = metadata['packages'][0]
    data = open(join(prjname, "__init__.py")).read()
    return re.search(expr, data).group(1)


metadata = get_metadata()
version = get_version(metadata)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = True
master_doc = 'index'
project = metadata['name']
copyright = '%s, %s' % (strftime("%Y"), metadata['maintainer'])
author = metadata['maintainer']
release = version
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'conf.py']
pygments_style = 'sphinx'
todo_include_todos = False
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}
