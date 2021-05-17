import io
import os
import re

from setuptools import find_packages
from setuptools import setup
from os import path


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="artefact_nca",
    version="0.1.0",
    url="https://github.com/kragniz/cookiecutter-pypackage-minimal",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Offical code for the paper: \"Growing 3D Artefacts and Functional Machines with Neural Cellular Automata\"",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'nbtlib',
        'torch',
        'hydra-core==1.1.0.rc1',
        'pydantic',
        'numpy',
        'attrs',
        'loguru',
        'tensorboard==2.4.0',
        'matplotlib',
        'einops',
        'tqdm',
        'grpcio',
        'test-evocraft-py',
        'protobuf',
        'click',
        'omegaconf',
        'ipython==7.16.1',
        'ipywidgets',
        'ipython-genutils==0.2.0',
        'wcwidth',
        'ptyprocess==0.6.0',
        'pytz',
        'requests',
        'google-auth',
        'google-auth-oauthlib',
        'oauthlib'
    ],

    include_package_data=True,

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
    ],
)
