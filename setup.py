from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

setup(
    name='bartpy',
    version='0.0.2',
    description='Bayesian Additive Regression Trees for Python',
    url='https://github.com/JakeColtman/bartpy',
    author='Jake Coltman',
    author_email='jakecoltman@sky.com',
    packages=['bartpy'],
    install_requires=[
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'sklearn',
        'statsmodels',
        'tqdm',
    ]
)


