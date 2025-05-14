from setuptools import setup, find_packages

setup(
    name="soniccypher",
    version="0.1.0",
    packages=find_packages(),  # include all folders with __init__.py
    py_modules=["run_pipline"]
)
