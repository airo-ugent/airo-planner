import setuptools
from setuptools import find_namespace_packages

setuptools.setup(
    packages=find_namespace_packages(exclude=["test", "build"]),
)
