"""
Data Engine Module
=================

Handles market data ingestion, normalization, storage, and retrieval.
Supports multiple data formats and provides efficient access to historical data.

Key Components:
- DataLoader: Loads data from various sources (CSV, Parquet, etc.)
- DataNormalizer: Standardizes data formats and timestamps
- DataStorage: Manages partitioned storage for efficient access
- DataValidator: Quality checks and validation
- ReplayEngine: Event generator for backtesting
"""

from .data_loader import DataLoader
from .data_normalizer import DataNormalizer
from .data_storage import DataStorage
from .data_validator import DataValidator
from .replay_engine import ReplayEngine

__all__ = [
    'DataLoader',
    'DataNormalizer', 
    'DataStorage',
    'DataValidator',
    'ReplayEngine'
]