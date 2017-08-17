from __future__ import unicode_literals

import re
from importlib import import_module
from os import getenv
from os.path import dirname, join, realpath
from time import strftime

import sphinx_rtd_theme

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser


def get_metadata():
    config = ConfigParser()
    dir_path = dirname(realpath(__file__))
    config.read(join(dir_path, '..', 'setup.cfg'))
    metadata = dict(config.items('metadata'))
    metadata['packages'] = eval(metadata['packages'])
    return metadata


def get_init_metadata(name):
    expr = re.compile(r"__%s__ *= *\"(.*)\"" % name)

    metadata = get_metadata()

    pkgname = metadata['packages'][0]
    data = open(join("..", pkgname, "__init__.py")).read()

    return re.search(expr, data).group(1).strip()


if getenv("READTHEDOCS", "False") == "True":

    prjname = getenv("READTHEDOCS_PROJECT", "unknown")
    pkgname = prjname.replace("-", "_")
    pkg = import_module(pkgname)

    project = pkg.__name__
    version = pkg.__version__
    author = pkg.__author__
else:
    project = get_init_metadata('name')
    version = get_init_metadata('version')
    author = get_init_metadata('author')

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
copyright = '%s, %s' % (strftime("%Y"), author)
release = version
language = "en"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'conf.py']
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
intersphinx_mapping = {
    'python': ('http://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}
