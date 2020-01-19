from distutils.core import setup
from pathlib import Path

import setuptools

dpath = Path(__file__).parent.absolute()

try:
    with open(dpath / 'requirements.txt') as f:
        # read recipe at https://www.reddit.com/r/Python/comments/3uzl2a/setuppy_requirementstxt_or_a_combination/
        requirements = f.read().splitlines()
except FileNotFoundError:
    pass

setup(name='sparse_ae',
      version='0.0.1',
      author='Mikhail Stepanov',
      author_email='mishc9@gmail.com',
      description='sparse autoencoder',
      packages=setuptools.find_packages(),
      include_package_data=True,
      python_requires='>=3.6.5',
      )
