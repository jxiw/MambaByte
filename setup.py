from glob import glob

from setuptools import setup, find_packages

setup(
    name="mambabyte",
    version="1.0",
    description="MambaByte: Token-free Selective State Space Model",
    author="Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, Alexander M Rush",
    author_email="tg352@cornell.edu",
    scripts=glob("scripts/*.py", recursive=True),
    py_modules=[],
    packages=find_packages(),
    python_requires=">=3.9",
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    install_requires=[
        "torch",
        "torchinfo",
        "prettytable",
        "einops",
        "transformers",
        "mamba-ssm",
    ],
    url="https://github.com/TushaarGVS/mambabyte",
)
