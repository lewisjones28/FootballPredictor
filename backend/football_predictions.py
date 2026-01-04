import logging
import os
import random
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from downloader import download_fixtures

# Constants
DEFAULT_LEAGUE = "epl"
DEFAULT_YEAR = 2025
DEFAULT_ITERATIONS_MULTIPLIER = 758
DEFAULT_TYPICAL_RESULT_PERCENTAGE = 0.9
DEFAULT_OUTPUT_DIRECTORY = 'backend/files/output'
DEFAULT_INPUT_DIRECTORY = 'backend/files/input'

# Prediction result types
HOME_WIN = 1
DRAW = 0
AWAY_WIN = -1

# Goal range constants
MIN_GOALS = 0
TYPICAL_MAX_GOALS = 2
HIGH_SCORE_MIN = 3
HIGH_SCORE_MAX = 5

# Model training constants
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Performance optimization constants
LOG_FREQUENCY = 100  # Log every N iterations instead of every iteration
BATCH_PROCESS_SIZE = 1000  # Process predictions in batches for memory efficiency
USE_MULTIPROCESSING = True  # Enable parallel processing for iterations
MAX_WORKERS = None  # None = use CPU count, or set specific number (e.g., 4, 8)

# Column names
COL_MATCH_NUMBER = 'Match Number'
COL_ROUND_NUMBER = 'Round Number'
COL_DATE = 'Date'
COL_LOCATION = 'Location'
COL_HOME_TEAM = 'Home Team'
COL_AWAY_TEAM = 'Away Team'
COL_RESULT = 'Result'
COL_PREDICTED = 'Predicted'
COL_HOME_FORM = 'Home Form'
COL_AWAY_FORM = 'Away Form'
COL_HOME_STRENGTH = 'Home Strength'
COL_AWAY_STRENGTH = 'Away Strength'

# File names
TRAINING_DATA_FILE = 'training_data.csv'
FINAL_PREDICTIONS_FILE = 'final_predictions.csv'

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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


class ModelTrainer:
    """Handles model training and data preparation."""

    def __init__(self, team_analyzer: TeamAnalyzer):
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


class ResultGenerator:
    """Handles dynamic result generation based on predictions."""

    @staticmethod
    def generate_dynamic_result(predicted_result: int, home_goals_avg: float, away_goals_avg: float,
                                typical_result_percentage: float = DEFAULT_TYPICAL_RESULT_PERCENTAGE) -> str:
        """Generate a dynamic result based on the predicted outcome."""
        # Ensure goal values do not go below zero
        home_goals_avg = max(home_goals_avg, MIN_GOALS)
        away_goals_avg = max(away_goals_avg, MIN_GOALS)

        if predicted_result == HOME_WIN:
            if random.random() < typical_result_percentage:
                home_goals = random.randint(max(MIN_GOALS, int(home_goals_avg - 1)), int(home_goals_avg + 2))
                away_goals = random.randint(MIN_GOALS, TYPICAL_MAX_GOALS)
            else:
                home_goals = random.randint(HIGH_SCORE_MIN, HIGH_SCORE_MAX)
                away_goals = random.randint(1, HIGH_SCORE_MIN)
            return f"{home_goals} - {away_goals}"

        elif predicted_result == DRAW:
            if random.random() < typical_result_percentage:
                home_goals = random.randint(max(MIN_GOALS, int(home_goals_avg - 1)), int(home_goals_avg + 1))
                away_goals = random.randint(max(MIN_GOALS, int(away_goals_avg - 1)), int(away_goals_avg + 1))
            else:
                home_goals = random.randint(TYPICAL_MAX_GOALS, 4)
                away_goals = random.randint(TYPICAL_MAX_GOALS, 4)
            return f"{home_goals} - {away_goals}"

        else:  # AWAY_WIN
            if random.random() < typical_result_percentage:
                home_goals = random.randint(MIN_GOALS, TYPICAL_MAX_GOALS)
                away_goals = random.randint(max(MIN_GOALS, int(away_goals_avg - 1)), int(away_goals_avg + 2))
            else:
                home_goals = random.randint(MIN_GOALS, 1)
                away_goals = random.randint(HIGH_SCORE_MIN, HIGH_SCORE_MAX)
            return f"{home_goals} - {away_goals}"


class PredictionEngine:
    """Main prediction engine that coordinates all components."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.team_analyzer = TeamAnalyzer()
        self.model_trainer = ModelTrainer(self.team_analyzer)
        self.result_generator = ResultGenerator()
        self.model = None
        self.predictions_memory: Dict[int, List[Dict]] = {}  # Store predictions in memory by round

    def load_data(self) -> pd.DataFrame:
        """Load and return the data from CSV file."""
        logging.info(f'Loading CSV file from {self.config.csv_file}.')
        return pd.read_csv(self.config.csv_file)

    def train_model(self, data: pd.DataFrame) -> float:
        """Train the machine learning model and return accuracy."""
        # Create training data
        training_df = self.model_trainer.create_training_data(data)

        # Extract features and labels
        X = training_df[[COL_HOME_FORM, COL_AWAY_FORM, COL_HOME_STRENGTH, COL_AWAY_STRENGTH]]
        y = training_df[COL_RESULT]

        # Save training data to file (this is acceptable as it's the final aggregated data)
        output_dir = f"{self.config.output_directory}/{self.config.league}/{self.config.year}"
        os.makedirs(output_dir, exist_ok=True)
        training_df.to_csv(f'{output_dir}/{TRAINING_DATA_FILE}', index=False)

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

        # Test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f'Model accuracy: {accuracy:.2f}')
        return accuracy

    def predict_match_result(self, home_team: str, away_team: str, data: pd.DataFrame) -> str:
        """Predict result for a single match."""
        logging.debug(f'Predicting match result for {home_team} vs {away_team}')

        # Calculate features
        home_form = self.team_analyzer.get_team_form(data, home_team, is_home=True)
        away_form = self.team_analyzer.get_team_form(data, away_team, is_home=False)
        home_strength = self.team_analyzer.calculate_overall_team_strength(data, home_team)
        away_strength = self.team_analyzer.calculate_overall_team_strength(data, away_team)

        # Prepare feature vector
        features = [home_form, away_form, home_strength, away_strength]
        prediction = self.model.predict([features])[0]

        if self.config.dynamic_results:
            logging.debug(f'Using dynamic results for predictions for {home_team} vs {away_team}')
            return self.result_generator.generate_dynamic_result(
                prediction, home_form, away_form, self.config.typical_result_percentage
            )
        else:
            # Static result logic
            if prediction == HOME_WIN:
                return f"{random.randint(TYPICAL_MAX_GOALS, 4)} - {random.randint(MIN_GOALS, 1)}"
            elif prediction == DRAW:
                return f"{random.randint(MIN_GOALS, TYPICAL_MAX_GOALS)} - {random.randint(MIN_GOALS, TYPICAL_MAX_GOALS)}"
            else:  # AWAY_WIN
                return f"{random.randint(MIN_GOALS, 1)} - {random.randint(TYPICAL_MAX_GOALS, 4)}"

    def run_predictions(self):
        """Run all predictions and store them in memory."""
        import time
        overall_start = time.time()

        logging.info("=" * 70)
        logging.info("FOOTBALL PREDICTIONS - STARTING")
        logging.info("=" * 70)

        # Load and cache data once
        logging.info("Step 1/4: Loading data...")
        data = self.load_data()
        logging.info(f"✓ Loaded {len(data)} rows from {self.config.csv_file}")

        # Train the model
        logging.info("Step 2/4: Training model...")
        accuracy = self.train_model(data)
        logging.info(f"✓ Model trained with {accuracy:.1%} accuracy")

        # Initialize predictions memory for each round
        logging.info("Step 3/4: Preparing prediction rounds...")
        for round_number in self.config.rounds_to_predict:
            self.predictions_memory[round_number] = []

        # Pre-filter matches to predict once (optimization)
        matches_to_predict = {}
        total_matches = 0
        for round_number in self.config.rounds_to_predict:
            matches_to_predict[round_number] = data.loc[
                (data[COL_ROUND_NUMBER] == round_number) & pd.isna(data[COL_RESULT])
                ].copy()
            num_matches = len(matches_to_predict[round_number])
            total_matches += num_matches
            logging.info(f"  Round {round_number}: {num_matches} matches to predict")

        logging.info(f"✓ Total: {total_matches} matches across {len(self.config.rounds_to_predict)} rounds")

        # Temporarily increase logging level to reduce I/O overhead during iterations
        original_log_level = logging.getLogger().level
        if self.config.iterations > 1000:
            logging.getLogger().setLevel(logging.INFO)
            logging.info("Large iteration count detected - reducing log verbosity for performance")

        # Run iterations with optimizations
        total_iterations = self.config.iterations
        logging.info("")
        logging.info("Step 4/4: Running predictions...")
        logging.info(f"Mode: {'PARALLEL' if USE_MULTIPROCESSING and total_iterations > 100 else 'SEQUENTIAL'}")
        logging.info("")

        # Check if multiprocessing should be used
        if USE_MULTIPROCESSING and total_iterations > 100:
            self._run_predictions_parallel(data, matches_to_predict, total_iterations)
        else:
            self._run_predictions_sequential(data, matches_to_predict, total_iterations)

        # Restore original logging level
        logging.getLogger().setLevel(original_log_level)

        total_time = time.time() - overall_start
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"ALL PREDICTIONS COMPLETED IN {total_time / 60:.2f} MINUTES")
        logging.info("=" * 70)

    def _run_predictions_sequential(self, data: pd.DataFrame, matches_to_predict: Dict, total_iterations: int):
        """Run predictions sequentially (original method)."""
        import time
        start_time = time.time()

        # Count total predictions to make
        total_matches = sum(len(matches_to_predict[rnd]) for rnd in self.config.rounds_to_predict)
        total_predictions = total_iterations * total_matches
        predictions_made = 0

        logging.info(
            f"Sequential processing: {total_matches} matches × {total_iterations} iterations = {total_predictions} total predictions")

        for iteration in range(total_iterations):
            # Log progress at intervals instead of every iteration
            if iteration % LOG_FREQUENCY == 0 or iteration == total_iterations - 1:
                elapsed = time.time() - start_time
                progress_pct = ((iteration + 1) / total_iterations * 100)

                # Estimate time remaining
                if iteration > 0:
                    avg_time_per_iter = elapsed / (iteration + 1)
                    remaining_iters = total_iterations - (iteration + 1)
                    eta_seconds = avg_time_per_iter * remaining_iters
                    eta_minutes = eta_seconds / 60

                    if eta_minutes < 1:
                        eta_str = f"{eta_seconds:.0f}s"
                    elif eta_minutes < 60:
                        eta_str = f"{eta_minutes:.1f}m"
                    else:
                        eta_hours = eta_minutes / 60
                        eta_str = f"{eta_hours:.1f}h"

                    logging.info(f'Progress: {iteration + 1}/{total_iterations} iterations ({progress_pct:.1f}%) | '
                                 f'Elapsed: {elapsed / 60:.1f}m | ETA: {eta_str} | '
                                 f'Speed: {(iteration + 1) / elapsed:.1f} iter/s')
                else:
                    logging.info(f'Progress: {iteration + 1}/{total_iterations} iterations ({progress_pct:.1f}%)')

            for round_number in self.config.rounds_to_predict:
                predictions_needed = matches_to_predict[round_number]

                if not predictions_needed.empty:
                    # Process all matches for this round at once
                    for _, row in predictions_needed.iterrows():
                        # Predict the result (caching helps here)
                        predicted_result = self.predict_match_result(
                            row[COL_HOME_TEAM], row[COL_AWAY_TEAM], data
                        )

                        # Store prediction in memory (minimal dictionary for memory efficiency)
                        self.predictions_memory[round_number].append({
                            'iteration': iteration + 1,
                            'home_team': row[COL_HOME_TEAM],
                            'away_team': row[COL_AWAY_TEAM],
                            'predicted_result': predicted_result,
                            'round_number': round_number,
                            'match_number': row[COL_MATCH_NUMBER]
                        })
                        predictions_made += 1

        total_time = time.time() - start_time
        logging.info(
            f"Sequential processing completed: {predictions_made} predictions in {total_time / 60:.2f} minutes "
            f"({predictions_made / total_time:.1f} predictions/second)")

    def _run_predictions_parallel(self, data: pd.DataFrame, matches_to_predict: Dict, total_iterations: int):
        """Run predictions in parallel using multiprocessing."""
        import time
        start_time = time.time()

        workers = MAX_WORKERS if MAX_WORKERS else cpu_count()
        total_matches = sum(len(matches_to_predict[rnd]) for rnd in self.config.rounds_to_predict)
        total_predictions = total_iterations * total_matches

        logging.info(f"╔═══════════════════════════════════════════════════════════════╗")
        logging.info(f"║ PARALLEL PROCESSING STARTED                                   ║")
        logging.info(f"╠═══════════════════════════════════════════════════════════════╣")
        logging.info(f"║ Workers:      {workers} CPU cores                             ║")
        logging.info(f"║ Iterations:   {total_iterations:,}                            ║")
        logging.info(f"║ Matches:      {total_matches} per iteration                   ║")
        logging.info(f"║ Total Preds:  {total_predictions:,}                           ║")
        logging.info(f"╚═══════════════════════════════════════════════════════════════╝")

        # Split iterations into chunks for parallel processing
        chunk_size = max(1,
                         total_iterations // (workers * 4))  # Create more chunks than workers for better load balancing
        iteration_ranges = []
        for i in range(0, total_iterations, chunk_size):
            end = min(i + chunk_size, total_iterations)
            iteration_ranges.append((i, end))

        logging.info(
            f"Split {total_iterations} iterations into {len(iteration_ranges)} chunks (avg {chunk_size} iterations/chunk)")

        # Prepare data for parallel processing
        worker_data = {
            'matches_to_predict': matches_to_predict,
            'rounds_to_predict': self.config.rounds_to_predict,
            'model': self.model,
            'team_analyzer': self.team_analyzer,
            'result_generator': self.result_generator,
            'config': self.config,
            'data': data
        }

        # Process chunks in parallel with progress tracking
        logging.info(f"Processing chunks across {workers} worker processes...")
        processing_start = time.time()

        # Use imap instead of map to get results as they complete
        with Pool(processes=workers) as pool:
            worker_func = partial(_process_iteration_chunk, worker_data=worker_data)
            chunk_results = []

            # Process with progress tracking
            for idx, result in enumerate(pool.imap(worker_func, iteration_ranges), 1):
                chunk_results.append(result)

                # Calculate progress
                chunks_completed = idx
                total_chunks = len(iteration_ranges)
                progress_pct = (chunks_completed / total_chunks) * 100
                elapsed = time.time() - processing_start

                # Estimate remaining time
                if chunks_completed > 0:
                    avg_time_per_chunk = elapsed / chunks_completed
                    remaining_chunks = total_chunks - chunks_completed
                    eta_seconds = avg_time_per_chunk * remaining_chunks

                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.0f}s"
                    else:
                        eta_str = f"{eta_seconds / 60:.1f}m"

                    # Calculate iterations completed so far
                    iters_per_chunk = total_iterations / total_chunks
                    iters_completed = int(chunks_completed * iters_per_chunk)

                    logging.info(f"  Chunk {chunks_completed}/{total_chunks} ({progress_pct:.1f}%) | "
                                 f"~{iters_completed:,}/{total_iterations:,} iterations | "
                                 f"Elapsed: {elapsed:.0f}s | ETA: {eta_str}")

        processing_time = time.time() - processing_start
        logging.info(
            f"✓ All chunks processed in {processing_time / 60:.2f} minutes ({total_iterations / processing_time:.1f} iter/s)")

        # Aggregate results from all workers
        logging.info("Aggregating results from parallel workers...")
        agg_start = time.time()
        total_preds_collected = 0

        for chunk_idx, chunk_predictions in enumerate(chunk_results):
            for round_number, predictions in chunk_predictions.items():
                self.predictions_memory[round_number].extend(predictions)
                total_preds_collected += len(predictions)

        agg_time = time.time() - agg_start
        total_time = time.time() - start_time

        logging.info(f"✓ Aggregated {total_preds_collected:,} predictions in {agg_time:.2f}s")
        logging.info(f"")
        logging.info(f"╔═══════════════════════════════════════════════════════════════╗")
        logging.info(f"║ PARALLEL PROCESSING COMPLETE                                  ║")
        logging.info(f"╠═══════════════════════════════════════════════════════════════╣")
        logging.info(f"║ Total Time:    {total_time / 60:.2f} minutes                    ║")
        logging.info(f"║ Speed:         {total_predictions / total_time:.1f} predictions/second            ║")
        logging.info(f"║ Throughput:    {total_iterations / total_time:.1f} iterations/second              ║")
        logging.info(f"╚═══════════════════════════════════════════════════════════════╝")

    def save_aggregated_predictions_from_memory(self):
        """Create and save final aggregated predictions directly from memory without individual iteration files."""
        from collections import Counter
        import time

        logging.info("")
        logging.info("=" * 70)
        logging.info("SAVING AGGREGATED PREDICTIONS")
        logging.info("=" * 70)

        start_time = time.time()

        # Load original data once for all rounds (optimization)
        logging.info("Loading original data for metadata...")
        original_data = pd.read_csv(self.config.csv_file)
        logging.info(f"✓ Loaded {len(original_data)} rows")

        for idx, round_number in enumerate(self.config.rounds_to_predict, 1):
            if round_number in self.predictions_memory:
                round_start = time.time()
                logging.info(f"")
                logging.info(f"Processing Round {round_number} ({idx}/{len(self.config.rounds_to_predict)})...")

                output_dir = f"{self.config.output_directory}/{self.config.league}/{self.config.year}/{round_number}"
                os.makedirs(output_dir, exist_ok=True)

                # Group predictions by match for aggregation
                logging.info(f"  Grouping {len(self.predictions_memory[round_number]):,} predictions by match...")
                match_predictions = {}

                for prediction in self.predictions_memory[round_number]:
                    match_key = (prediction['home_team'], prediction['away_team'], prediction['round_number'])
                    if match_key not in match_predictions:
                        match_predictions[match_key] = {
                            'results': [],
                            'match_info': prediction
                        }
                    match_predictions[match_key]['results'].append(prediction['predicted_result'])

                logging.info(f"  ✓ Grouped into {len(match_predictions)} unique matches")

                # Create final aggregated predictions
                logging.info(f"  Aggregating results (finding most common predictions)...")
                final_predictions = []

                for match_key, match_data in match_predictions.items():
                    home_team, away_team, round_num = match_key

                    # Get original match information
                    original_row = original_data[
                        (original_data[COL_ROUND_NUMBER] == round_num) &
                        (original_data[COL_HOME_TEAM] == home_team) &
                        (original_data[COL_AWAY_TEAM] == away_team)
                        ].iloc[0]

                    # Find most common result
                    result_counts = Counter(match_data['results'])
                    most_common_result = result_counts.most_common(1)[0][0]

                    # Create final prediction row
                    final_row = {
                        COL_MATCH_NUMBER: match_data['match_info']['match_number'],
                        COL_ROUND_NUMBER: round_num,
                        COL_DATE: original_row[COL_DATE] if COL_DATE in original_row else '',
                        COL_LOCATION: original_row[COL_LOCATION] if COL_LOCATION in original_row else '',
                        COL_HOME_TEAM: home_team,
                        COL_AWAY_TEAM: away_team,
                        COL_RESULT: most_common_result,
                        COL_PREDICTED: True
                    }
                    final_predictions.append(final_row)

                # Save final aggregated predictions
                logging.info(f"  Saving to {FINAL_PREDICTIONS_FILE}...")
                final_df = pd.DataFrame(final_predictions)
                final_df = final_df.sort_values(COL_MATCH_NUMBER).reset_index(drop=True)
                final_df.to_csv(f"{output_dir}/{FINAL_PREDICTIONS_FILE}", index=False)

                round_time = time.time() - round_start
                logging.info(
                    f"  ✓ Round {round_number} saved in {round_time:.2f}s → {output_dir}/{FINAL_PREDICTIONS_FILE}")

        total_time = time.time() - start_time
        logging.info("")
        logging.info(f"✓ All {len(self.config.rounds_to_predict)} rounds saved in {total_time:.2f}s")
        logging.info("=" * 70)

    def save_final_predictions(self):
        """Deprecated: Use save_aggregated_predictions_from_memory() instead."""
        logging.warning(
            "save_final_predictions() is deprecated. Individual iteration files are no longer saved to preserve disk space.")
        pass

    def get_predictions_summary(self) -> Dict:
        """Get a summary of all predictions stored in memory."""
        summary = {}
        for round_number, predictions in self.predictions_memory.items():
            summary[round_number] = {
                'total_predictions': len(predictions),
                'iterations': len(set(p['iteration'] for p in predictions)),
                'matches': len(set((p['home_team'], p['away_team']) for p in predictions))
            }
        return summary


# Worker function for multiprocessing (must be at module level)
def _process_iteration_chunk(iteration_range: tuple, worker_data: dict) -> Dict[int, List[Dict]]:
    """Process a chunk of iterations in parallel. This function runs in a separate process."""
    start_iter, end_iter = iteration_range
    matches_to_predict = worker_data['matches_to_predict']
    rounds_to_predict = worker_data['rounds_to_predict']
    model = worker_data['model']
    team_analyzer = worker_data['team_analyzer']
    result_generator = worker_data['result_generator']
    config = worker_data['config']
    data = worker_data['data']

    # Initialize local predictions storage
    local_predictions = {round_num: [] for round_num in rounds_to_predict}

    # Process iterations in this chunk
    for iteration in range(start_iter, end_iter):
        for round_number in rounds_to_predict:
            predictions_needed = matches_to_predict[round_number]

            if not predictions_needed.empty:
                for _, row in predictions_needed.iterrows():
                    # Calculate features
                    home_form = team_analyzer.get_team_form(data, row[COL_HOME_TEAM], is_home=True)
                    away_form = team_analyzer.get_team_form(data, row[COL_AWAY_TEAM], is_home=False)
                    home_strength = team_analyzer.calculate_overall_team_strength(data, row[COL_HOME_TEAM])
                    away_strength = team_analyzer.calculate_overall_team_strength(data, row[COL_AWAY_TEAM])

                    # Predict
                    features = [home_form, away_form, home_strength, away_strength]
                    prediction = model.predict([features])[0]

                    # Generate result
                    if config.dynamic_results:
                        predicted_result = result_generator.generate_dynamic_result(
                            prediction, home_form, away_form, config.typical_result_percentage
                        )
                    else:
                        if prediction == HOME_WIN:
                            predicted_result = f"{random.randint(TYPICAL_MAX_GOALS, 4)} - {random.randint(MIN_GOALS, 1)}"
                        elif prediction == DRAW:
                            predicted_result = f"{random.randint(MIN_GOALS, TYPICAL_MAX_GOALS)} - {random.randint(MIN_GOALS, TYPICAL_MAX_GOALS)}"
                        else:
                            predicted_result = f"{random.randint(MIN_GOALS, 1)} - {random.randint(TYPICAL_MAX_GOALS, 4)}"

                    # Store prediction
                    local_predictions[round_number].append({
                        'iteration': iteration + 1,
                        'home_team': row[COL_HOME_TEAM],
                        'away_team': row[COL_AWAY_TEAM],
                        'predicted_result': predicted_result,
                        'round_number': round_number,
                        'match_number': row[COL_MATCH_NUMBER]
                    })

    return local_predictions


def main():
    """Main function to run the football predictions."""
    # Set up configuration
    config = ConfigurationManager.setup_configuration()

    # Create and run prediction engine
    engine = PredictionEngine(config)
    engine.run_predictions()

    # Print summary
    summary = engine.get_predictions_summary()
    logging.info(f"Predictions summary: {summary}")

    # Save final aggregated predictions directly from memory (no individual iteration files)
    engine.save_aggregated_predictions_from_memory()

    logging.info("All predictions completed and final aggregated results saved.")


if __name__ == "__main__":
    main()
