from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'pandas==0.24.2',
    'scikit-learn==0.20.4'
]

setup(
    name='TaxiFareModel',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Taxi Fare Prediction Pipeline'
)


