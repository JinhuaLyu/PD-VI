from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pd-vi",
    version="0.1.0",
    author="Research Team",
    description="Primal-Dual Variational Inference for Gaussian Mixture Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="experiments"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4",
        "matplotlib>=3.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
