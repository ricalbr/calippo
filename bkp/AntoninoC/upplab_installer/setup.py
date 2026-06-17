from setuptools import setup, find_packages

setup(
    name='upplab',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='A collection of useful Python scripts to work with ICE, AQUA, CAPABLE and other labs.',
    author='Niki Di Giano',
    install_requires=[
    'numpy',
    'scipy',
    'matplotlib',
    'pyvisa',
    'ThorlabsPM100',
    'qontrol'
],
)