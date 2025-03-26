from setuptools import setup, find_packages

setup(
    name="critters-neuro-evolution",
    version="0.1.0",
    description="A 2D simulation of triangular critters with neural network-controlled behavior and evolution",
    author="Gael Reinaudi",
    packages=find_packages(),
    install_requires=[
        "pygame>=2.0.0",
        "numpy>=1.19.0",
        "neat-python>=0.92",
    ],
    python_requires=">=3.6",
) 