import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.4.0'
PACKAGE_NAME = 'linear-tree'
AUTHOR = 'Marco Cerliani, Ethan Sunshine'
AUTHOR_EMAIL = 'emsunshine@andrew.cmu.edu'
URL = 'https://github.com/emsunshine/linear-tree'

LICENSE = 'MIT'
DESCRIPTION = 'A python library to build Model Trees with Linear Models at the leaves.'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    'scikit-learn>=0.24.2',
    'numpy<2.0.0',
    'scipy'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      python_requires='>=3',
      packages=find_packages()
      )
