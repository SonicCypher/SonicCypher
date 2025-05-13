from setuptools import setup, find_packages

setup(
    name="soniccypher",
    version="0.1.0",
    packages=find_packages(include=["Res2Net*", "Model*", "Decision_Fusion*", "Preprocessing*"]),
)
