#!/usr/bin/python3

import setuptools
import os
import sys
sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'src'
))
import wcsh


setuptools.setup(
        name='oracle',
        version=wcsh.__version__,
        description="Functional test of omega oracle frontend",
        author='Cameron Allen',
        author_email='cameron@cameronpallen.com',
        packages=setuptools.find_packages('src'),
        package_dir={'': 'src'},
        install_requires=[
            'aiohttp==2.1.0',
            'numpy==1.12.1',
            'pandas==0.20.2',
            'mock==2.0.0',
        ],
)
