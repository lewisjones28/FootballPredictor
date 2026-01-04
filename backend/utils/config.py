import os
from dataclasses import dataclass
from typing import List

import pandas as pd

from backend.io.downloader import download_fixtures
from backend.utils.constants import (
    DEFAULT_YEAR, DEFAULT_ITERATIONS_MULTIPLIER,
    DEFAULT_TYPICAL_RESULT_PERCENTAGE, DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_INPUT_DIRECTORY, COL_ROUND_NUMBER
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
        # Allow selecting a single league or 'all' (process all leagues present under input dir)
        league_input = input(
            f"Enter a league (e.g. epl, championship, la-liga, serie-a, bundesliga) (default: 'all'): "
        ).strip()
        if not league_input:
            league_input = 'all'
        year = int(ConfigurationManager.get_input("Enter a year", DEFAULT_YEAR))
        # Default behavior changed to 'all' rounds when left empty
        rounds_to_predict_input = input(
            "Enter the rounds to predict (default: 'all'): \n"
            " - Leave empty for ALL rounds from the CSV\n"
            " - Enter 'current and next' to use the previous behaviour\n"
            " - Or enter space-separated round numbers: "
        ).strip()
        iterations = int(
            ConfigurationManager.get_input("Enter the number of iterations", DEFAULT_ITERATIONS_MULTIPLIER))

        # Determine list of leagues to process
        # Prefer a fixed list of common leagues when the user selects 'all'
        KNOWN_LEAGUES = ['epl', 'championship', 'la-liga', 'serie-a', 'bundesliga']
        leagues_to_process = []
        if league_input.lower() in ('all', 'all leagues'):
            # Use the canonical known leagues first so we process them even if
            # the local input directories don't yet exist (the downloader will create them).
            leagues_to_process = KNOWN_LEAGUES.copy()
            # Also include any discovered directories that are not in the known list
            input_root = DEFAULT_INPUT_DIRECTORY
            if os.path.exists(input_root):
                for entry in os.listdir(input_root):
                    entry_path = os.path.join(input_root, entry)
                    if os.path.isdir(entry_path) and entry not in leagues_to_process:
                        leagues_to_process.append(entry)
        else:
            leagues_to_process = [league_input]

        # Build a PredictionConfig for each league
        configs = []
        for league in leagues_to_process:
            # Create the input path and ensure fixtures exist (download if needed)
            path = f'{DEFAULT_INPUT_DIRECTORY}/{league}'
            os.makedirs(path, exist_ok=True)
            try:
                download_fixtures(league, year)
            except Exception:
                # Non-fatal: proceed even if download fails; assume file may already exist
                pass

            csv_file = f'{DEFAULT_INPUT_DIRECTORY}/{league}/{year}.csv'

            # Determine rounds to predict for this league/csv
            if not rounds_to_predict_input or rounds_to_predict_input.lower() in ("all", "all rounds"):
                # Default to all rounds present in the CSV file (robust conversion)
                data = pd.read_csv(csv_file)
                raw_rounds = data[COL_ROUND_NUMBER].dropna().unique()
                rounds_clean: List[int] = []
                for r in raw_rounds:
                    try:
                        rounds_clean.append(int(r))
                    except (ValueError, TypeError):
                        # Skip non-integer round identifiers
                        continue
                rounds_to_predict = sorted(set(rounds_clean))
            elif rounds_to_predict_input.lower() == "current and next":
                data = pd.read_csv(csv_file)
                from backend.data.loader import DataManager
                rounds_to_predict = DataManager.get_current_and_next_rounds(data)
            else:
                rounds_to_predict = list(map(int, rounds_to_predict_input.split()))

            cfg = PredictionConfig(
                league=league,
                year=year,
                iterations=iterations,
                output_directory=DEFAULT_OUTPUT_DIRECTORY,
                dynamic_results=True,
                typical_result_percentage=DEFAULT_TYPICAL_RESULT_PERCENTAGE,
                csv_file=csv_file,
                rounds_to_predict=rounds_to_predict
            )
            configs.append(cfg)

        # If only one config, return the single object for backwards compatibility
        if len(configs) == 1:
            return configs[0]
        return configs
