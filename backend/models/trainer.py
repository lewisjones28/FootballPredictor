import pandas as pd

from backend.utils.constants import (
    COL_HOME_TEAM, COL_AWAY_TEAM, COL_HOME_FORM, COL_AWAY_FORM,
    COL_HOME_STRENGTH, COL_AWAY_STRENGTH, COL_ROUND_NUMBER, COL_MATCH_NUMBER,
    COL_RESULT, HOME_WIN, AWAY_WIN, DRAW
)


class ModelTrainer:
    """Handles model training and data preparation."""

    def __init__(self, team_analyzer):
        self.team_analyzer = team_analyzer

    def create_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create training data using historical results and features."""
        training_data = []

        # Ensure the data is sorted by Round Number and Match Number
        data = data.sort_values(by=[COL_ROUND_NUMBER, COL_MATCH_NUMBER])

        for idx, match in data.iterrows():
            if pd.notna(match[COL_RESULT]):  # Only consider matches with results for training
                home_team = match[COL_HOME_TEAM]
                away_team = match[COL_AWAY_TEAM]

                # Calculate incremental features
                home_form = self.team_analyzer.get_team_form_incremental(data, home_team, True, idx)
                away_form = self.team_analyzer.get_team_form_incremental(data, away_team, False, idx)
                home_strength = self.team_analyzer.get_team_strength_incremental(data, home_team, idx)
                away_strength = self.team_analyzer.get_team_strength_incremental(data, away_team, idx)

                # Get the actual result
                home_goals = int(match[COL_RESULT].split(' - ')[0])
                away_goals = int(match[COL_RESULT].split(' - ')[1])

                # Create feature vector
                row = {
                    COL_HOME_TEAM: home_team,
                    COL_AWAY_TEAM: away_team,
                    COL_HOME_FORM: home_form,
                    COL_AWAY_FORM: away_form,
                    COL_HOME_STRENGTH: home_strength,
                    COL_AWAY_STRENGTH: away_strength,
                    COL_ROUND_NUMBER: match[COL_ROUND_NUMBER],
                    COL_MATCH_NUMBER: match[COL_MATCH_NUMBER]
                }

                # Target: 1 for home win, 0 for draw, -1 for away win
                if home_goals > away_goals:
                    row[COL_RESULT] = HOME_WIN
                elif home_goals < away_goals:
                    row[COL_RESULT] = AWAY_WIN
                else:
                    row[COL_RESULT] = DRAW

                training_data.append(row)

        # Create DataFrame and group by teams
        training_df = pd.DataFrame(training_data)
        grouped_training_data = training_df.groupby([COL_HOME_TEAM, COL_AWAY_TEAM], as_index=False).agg({
            COL_HOME_FORM: 'mean',
            COL_AWAY_FORM: 'mean',
            COL_HOME_STRENGTH: 'mean',
            COL_AWAY_STRENGTH: 'mean',
            COL_RESULT: 'mean',
            COL_ROUND_NUMBER: 'first',
            COL_MATCH_NUMBER: 'first'
        })

        # Sort by chronological order
        grouped_training_data.sort_values(by=[COL_ROUND_NUMBER, COL_MATCH_NUMBER], ascending=True, inplace=True)
        return grouped_training_data
