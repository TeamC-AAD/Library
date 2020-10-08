import os
from setuptools import setup, find_packages

with open("README.md", "r") as help_file:
    long_description = help_file.read()

setup(
    name="GTestAlot",
    version=1.0,
    description="Genes and genes",
    long_description=long_description,
    long_description_content_type="text/markdown"
)
