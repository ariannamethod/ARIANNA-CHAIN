from setuptools import setup

setup(
    name="arianna-chain",
    version="0.1.0",
    description="Arianna-C core engine",
    py_modules=["arianna_chain"],
    install_requires=["torch", "tokenizers"],
)
