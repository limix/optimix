import re
import sys
from os import chdir, getcwd
from os.path import abspath, dirname, join

from setuptools import setup

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser


class setup_folder(object):
    def __init__(self):
        self._old_path = None

    def __enter__(self):
        src_path = dirname(abspath(sys.argv[0]))
        self._old_path = getcwd()
        chdir(src_path)
        sys.path.insert(0, src_path)

    def __exit__(self, *_):
        del sys.path[0]
        chdir(self._old_path)


def set_names(metadata):
    pkgname = metadata['name']
    prjname = pkgname.replace('-', '_')
    metadata['packages'] = [prjname]


def set_version(metadata):
    expr = re.compile(r"__version__ *= *\"(.*)\"")
    prjname = metadata['packages'][0]
    data = open(join(prjname, "__init__.py")).read()
    metadata['version'] = re.search(expr, data).group(1)


def make_list(metadata, name):
    if name in metadata:
        metadata[name] = metadata[name].strip().split('\n')


def set_long_description(metadata):
    df = metadata['description_file']
    metadata['long_description'] = open(df).read()
    del metadata['description_file']


def setup_package():
    with setup_folder():

        config = ConfigParser()
        config.read('setup.cfg')
        metadata = dict(config.items('metadata'))

        set_names(metadata)
        set_version(metadata)
        make_list(metadata, 'classifiers')
        make_list(metadata, 'keywords')
        set_long_description(metadata)

        setup(**metadata)


if __name__ == '__main__':
    setup_package()
