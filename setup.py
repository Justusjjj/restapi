#!/usr/bin/env python3
"""
Setup script for Advanced Forex Trading Bot
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="advanced-forex-bot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Forex Trading Bot with ML, ICT Analysis, and Mathematical Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-forex-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "advanced": [
            "yfinance>=0.2.18",
            "tranchpy>=0.1.0",
            "pandas-datareader>=0.10.0",
            "fredapi>=0.5.0",
            "investpy>=1.0.8",
            "scikit-learn>=1.3.0",
            "tensorflow>=2.13.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
            "torch>=2.0.0",
            "statsmodels>=0.14.0",
            "arch>=6.2.0",
        ],
        "web": [
            "flask>=2.3.0",
            "flask-socketio>=5.3.0",
            "dash>=2.11.0",
            "plotly>=5.15.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forex-bot=final_forex_bot:main",
            "simple-forex-bot=simple_forex_bot:main",
        ],
    },
    keywords="forex, trading, bot, machine learning, ICT, algorithmic trading",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/advanced-forex-bot/issues",
        "Source": "https://github.com/yourusername/advanced-forex-bot",
        "Documentation": "https://github.com/yourusername/advanced-forex-bot#readme",
    },
)