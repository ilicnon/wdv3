from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="wd_timm",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)