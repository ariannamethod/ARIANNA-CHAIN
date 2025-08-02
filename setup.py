from setuptools import find_packages, setup

setup(
    name="arianna-c",
    version="0.1.0",
    description="Arianna-C core engine",
    packages=find_packages(),
    install_requires=["torch", "tokenizers"],
)
