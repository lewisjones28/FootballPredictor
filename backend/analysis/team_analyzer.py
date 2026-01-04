import logging
from typing import Dict

import pandas as pd

from backend.utils.constants import (
    COL_HOME_TEAM, COL_AWAY_TEAM, COL_RESULT
)


class TeamAnalyzer:
    """Handles team performance analysis and caching."""

    def __init__(self):
        self.team_form_cache: Dict[str, float] = {}
        self.team_strength_cache: Dict[str, float] = {}
        self.incremental_form_cache: Dict[str, Dict[int, float]] = {}
        self.incremental_strength_cache: Dict[str, Dict[int, float]] = {}

    def get_team_form(self, data: pd.DataFrame, team: str, is_home: bool = True) -> float:
        """Calculate the average goal difference for a team, with caching."""
        cache_key = f"{team}_home" if is_home else f"{team}_away"

        if cache_key in self.team_form_cache:
            logging.debug(f'Using cached form for {cache_key}')
            return self.team_form_cache[cache_key]

        # Calculate the form
        matches = data[data[COL_HOME_TEAM] == team] if is_home else data[data[COL_AWAY_TEAM] == team]

        results = []
        for _, match in matches.iterrows():
            if pd.notna(match[COL_RESULT]):
                home_goals, away_goals = map(int, match[COL_RESULT].split(' - '))
                if is_home:
                    results.append(home_goals - away_goals)
                else:
                    results.append(away_goals - home_goals)

        avg_goal_diff = sum(results) / len(results) if results else 0
        logging.debug(f'Team form calculated for {cache_key}: {avg_goal_diff}')

        self.team_form_cache[cache_key] = avg_goal_diff
        return avg_goal_diff

    def calculate_overall_team_strength(self, data: pd.DataFrame, team: str) -> float:
        """Calculate the strength of a team based on various metrics."""
        if team in self.team_strength_cache:
            logging.debug(f'Using cached overall team strength for {team}')
            return self.team_strength_cache[team]

        # Calculate team form (home and away)
        home_form = self.get_team_form(data, team, is_home=True)
        away_form = self.get_team_form(data, team, is_home=False)

        # Calculate average goals scored and conceded
        matches = data[(data[COL_HOME_TEAM] == team) | (data[COL_AWAY_TEAM] == team)]
        goals_scored = 0
        goals_conceded = 0

        for _, match in matches.iterrows():
            if pd.notna(match[COL_RESULT]):
                home_goals, away_goals = map(int, match[COL_RESULT].split(' - '))
                if match[COL_HOME_TEAM] == team:
                    goals_scored += home_goals
                    goals_conceded += away_goals
                else:
                    goals_scored += away_goals
                    goals_conceded += home_goals

        avg_goals_scored = goals_scored / len(matches) if len(matches) > 0 else 0
        avg_goals_conceded = goals_conceded / len(matches) if len(matches) > 0 else 0

        # Combine factors to calculate strength
        strength = home_form + away_form + avg_goals_scored - avg_goals_conceded
        self.team_strength_cache[team] = strength
        logging.debug(f'Team strength for {team}: {strength}')
        return strength

    def get_team_strength_incremental(self, data: pd.DataFrame, team: str, idx: int) -> float:
        """Calculate the incremental strength for a team up to the given match index."""
        cache_key = f"{team}_strength"

        if cache_key in self.incremental_strength_cache and idx in self.incremental_strength_cache[cache_key]:
            return self.incremental_strength_cache[cache_key][idx]

        # Filter matches involving the team up to index `idx`
        matches = data.loc[(data[COL_HOME_TEAM] == team) | (data[COL_AWAY_TEAM] == team)].iloc[:idx + 1]

        points = 0
        total_matches = 0
        goal_diff_sum = 0

        for _, match in matches.iterrows():
            if pd.notna(match[COL_RESULT]):
                home_goals, away_goals = map(int, match[COL_RESULT].split(' - '))
                if match[COL_HOME_TEAM] == team:
                    goal_diff = home_goals - away_goals
                    goal_diff_sum += goal_diff
                    points += 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
                else:
                    goal_diff = away_goals - home_goals
                    goal_diff_sum += goal_diff
                    points += 3 if away_goals > home_goals else 1 if away_goals == home_goals else 0
                total_matches += 1

        avg_goal_difference = goal_diff_sum / total_matches if total_matches else 0

        # Cache the strength metrics
        if cache_key not in self.incremental_strength_cache:
            self.incremental_strength_cache[cache_key] = {}

        self.incremental_strength_cache[cache_key][idx] = avg_goal_difference
        return avg_goal_difference

    def get_team_form_incremental(self, data: pd.DataFrame, team: str, is_home: bool, idx: int) -> float:
        """Calculate the incremental form for a team up to the given match index."""
        cache_key = f"{team}_home" if is_home else f"{team}_away"

        if cache_key in self.incremental_form_cache and idx in self.incremental_form_cache[cache_key]:
            return self.incremental_form_cache[cache_key][idx]

        # Calculate form up to the match with index `idx`
        matches = data[data[COL_HOME_TEAM] == team].iloc[:idx + 1] if is_home else data[
            data[COL_AWAY_TEAM] == team].iloc[:idx + 1]

        results = []
        for _, match in matches.iterrows():
            if pd.notna(match[COL_RESULT]):
                home_goals, away_goals = map(int, match[COL_RESULT].split(' - '))
                if is_home:
                    results.append(home_goals - away_goals)
                else:
                    results.append(away_goals - home_goals)

        avg_goal_diff = sum(results) / len(results) if results else 0

        # Cache the form at this index
        if cache_key not in self.incremental_form_cache:
            self.incremental_form_cache[cache_key] = {}

        self.incremental_form_cache[cache_key][idx] = avg_goal_diff
        return avg_goal_diff
