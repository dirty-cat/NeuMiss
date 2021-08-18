#!/usr/bin/env python

from pathlib import Path
from setuptools import setup, find_packages

version_file = Path(__file__) / 'neumiss' / 'VERSION.txt'
with open(version_file) as fh:
    VERSION = fh.read().strip()

description_file = Path(__file__) / 'README.rst'
with open(description_file) as fh:
    DESCRIPTION = fh.read()


if __name__ == '__main__':
    setup(
        name='neumiss',
        version=VERSION,
        author='Marine LE MORVAN',
        author_email='marine.le-morvan@inria.fr',
        description="NeuMiss is a neural network architecture aimed at handling missing values, usually used as a preprocessing layer.",
        long_description=DESCRIPTION,
        license='BSD',
        classifiers=[
            'Development Status:: 2 - Pre - Alpha',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering',
            'Topic :: Software Development :: Libraries',
        ],
        platforms='any',
        packages=find_packages(),
        install_requires=[
            'scikit-learn>=0.21',
            'numpy>=1.16',
            'scipy>=1.2',
            'requests',
            'joblib',
        ],
    )
