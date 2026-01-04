"""Backend package for FootballPredictor.

This package exposes the main backend modules. Keeping a small init to make
imports explicit and to allow `from backend import ...` usage.
"""

from .downloader import download_fixtures
from .football_predictions import (
    PredictionConfig,
    ConfigurationManager,
    DataManager,
    TeamAnalyzer,
    ModelTrainer,
    ResultGenerator,
    PredictionEngine,
    main,
)

__all__ = [
    'download_fixtures',
    'PredictionConfig',
    'ConfigurationManager',
    'DataManager',
    'TeamAnalyzer',
    'ModelTrainer',
    'ResultGenerator',
    'PredictionEngine',
    'main',
]

