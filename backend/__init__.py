"""Backend package for FootballPredictor.

This package exposes the main backend modules. Keeping a small init to make
imports explicit and to allow `from backend import ...` usage.
"""

from backend.analysis.team_analyzer import TeamAnalyzer
from backend.data.loader import DataManager
from backend.io.downloader import download_fixtures
from backend.models.engine import ResultGenerator, PredictionEngine, main
from backend.models.trainer import ModelTrainer
from backend.utils.config import PredictionConfig, ConfigurationManager

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
