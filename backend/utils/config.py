import os
from dataclasses import dataclass
from typing import List

import pandas as pd

from backend.io.downloader import download_fixtures
from backend.utils.constants import (
    DEFAULT_LEAGUE, DEFAULT_YEAR, DEFAULT_ITERATIONS_MULTIPLIER,
    DEFAULT_TYPICAL_RESULT_PERCENTAGE, DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_INPUT_DIRECTORY
)


@dataclass
class PredictionConfig:
    """Configuration class for football predictions."""
    league: str
    year: int
    iterations: int
    output_directory: str
    dynamic_results: bool
    typical_result_percentage: float
    csv_file: str
    rounds_to_predict: List[int]


class ConfigurationManager:
    """Handles user input and configuration setup."""

    @staticmethod
    def get_input(prompt: str, default_value) -> str:
        """Prompt the user for input with a default value."""
        user_input = input(f"{prompt} (default: {default_value}): ")
        return user_input if user_input else str(default_value)

    @staticmethod
    def setup_configuration() -> PredictionConfig:
        """Set up configuration from user input."""
        league = ConfigurationManager.get_input("Enter a league (e.g. epl, championship, la-liga, serie-a)",
                                                DEFAULT_LEAGUE)
        year = int(ConfigurationManager.get_input("Enter a year", DEFAULT_YEAR))
        rounds_to_predict_input = input("Enter the rounds to predict (default: 'current and next'): ").strip()
        iterations = int(
            ConfigurationManager.get_input("Enter the number of iterations", DEFAULT_ITERATIONS_MULTIPLIER))

        # Create the path and download fixtures
        path = f'{DEFAULT_INPUT_DIRECTORY}/{league}'
        os.makedirs(path, exist_ok=True)
        download_fixtures(league, year)

        csv_file = f'{DEFAULT_INPUT_DIRECTORY}/{league}/{year}.csv'

        # Determine rounds to predict
        if not rounds_to_predict_input or rounds_to_predict_input.lower() == "current and next":
            data = pd.read_csv(csv_file)
            from backend.data.loader import DataManager
            rounds_to_predict = DataManager.get_current_and_next_rounds(data)
        else:
            rounds_to_predict = list(map(int, rounds_to_predict_input.split()))

        return PredictionConfig(
            league=league,
            year=year,
            iterations=iterations,
            output_directory=DEFAULT_OUTPUT_DIRECTORY,
            dynamic_results=True,
            typical_result_percentage=DEFAULT_TYPICAL_RESULT_PERCENTAGE,
            csv_file=csv_file,
            rounds_to_predict=rounds_to_predict
        )
