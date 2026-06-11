"""
setup.py
========
Metadata del paquete AMD Time-Series.
Permite instalar el proyecto con: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="amd-timeseries",
    version="1.0.0",
    description="Comparativa de RNN vs LSTM para predicción del precio de cierre de AMD.",
    author="Christopher Andres Obando Rivera",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "yfinance>=0.2.40",
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "scikit-learn>=1.5.2",
        "tensorflow>=2.17.0",
        "matplotlib>=3.9.2",
        "python-dotenv>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "pytest-cov>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "amd-pipeline=main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
