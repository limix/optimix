import os
import sys

from setuptools import find_packages, setup

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (OSError, IOError, ImportError):
    long_description = open('README.md').read()


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
    pytest_runner = ['pytest-runner'] if needs_pytest else []

    setup_requires = [] + pytest_runner
    install_requires = [
        'scipy>=0.17', 'numpy>=1.9', 'ndarray-listener>=1.0.18',
        'brent-search>=1.0.17', 'tqdm'
    ]
    tests_require = ['pytest>=3']

    metadata = dict(
        name='optimix',
        version='1.1.8',
        maintainer="Danilo Horta",
        maintainer_email="horta@ebi.ac.uk",
        description="Abstract function optimisation.",
        long_description=long_description,
        license="MIT",
        url='https://github.com/glimix/optimix',
        packages=find_packages(),
        zip_safe=True,
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        include_package_data=True,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ], )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)


if __name__ == '__main__':
    setup_package()
