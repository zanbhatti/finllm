from setuptools import setup, find_packages

setup(
    name="finllm",
    version="0.1.0",
    packages=find_packages(include=["finllm", "finllm.*"]),
    install_requires=[
        # put dependencies here, e.g. "numpy", "pandas", "QuantLib"
    ],
    python_requires=">=3.9",
)
