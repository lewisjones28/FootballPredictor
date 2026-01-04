from typing import List

import pandas as pd

from backend.utils.constants import COL_RESULT, COL_ROUND_NUMBER


class DataManager:
    """Handles data loading and basic operations."""

    @staticmethod
    def get_current_and_next_rounds(data: pd.DataFrame) -> List[int]:
        """Identify the current and next round based on missing results."""
        missing_results = data[data[COL_RESULT].isna()]

        if missing_results.empty:
            return []

        current_round = missing_results[COL_ROUND_NUMBER].min()
        next_round = current_round + 1
        return [current_round, next_round]

    @staticmethod
    def update_rounds_to_predict(data: pd.DataFrame, rounds_to_predict: List[int]) -> List[int]:
        """Update rounds_to_predict based on missing results."""
        missing_rounds = data[data[COL_RESULT].isna()][COL_ROUND_NUMBER].unique()
        return [round_number for round_number in rounds_to_predict if round_number in missing_rounds]
