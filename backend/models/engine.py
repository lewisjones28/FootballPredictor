import logging
import os
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from backend.analysis.team_analyzer import TeamAnalyzer
from backend.models.trainer import ModelTrainer
from backend.utils.constants import (
    DEFAULT_TYPICAL_RESULT_PERCENTAGE, COL_MATCH_NUMBER, COL_ROUND_NUMBER, COL_DATE, COL_LOCATION,
    COL_HOME_TEAM, COL_AWAY_TEAM, COL_RESULT, COL_PREDICTED,
    COL_HOME_FORM, COL_AWAY_FORM, COL_HOME_STRENGTH, COL_AWAY_STRENGTH,
    TRAINING_DATA_FILE, FINAL_PREDICTIONS_FILE, HOME_WIN, DRAW, MIN_GOALS, TYPICAL_MAX_GOALS, HIGH_SCORE_MIN,
    HIGH_SCORE_MAX,
    TEST_SIZE, RANDOM_STATE, LOG_FREQUENCY, USE_MULTIPROCESSING, MAX_WORKERS
)


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

    def __init__(self, config):
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

                # Keep track of actual rows we've already appended to avoid duplicates
                appended_actual_keys = set()

                # First, build aggregated predicted results (if any predictions exist)
                for match_key, match_data in match_predictions.items():
                    home_team, away_team, round_num = match_key

                    # Get original match information
                    original_row = original_data[
                        (original_data[COL_ROUND_NUMBER] == round_num) &
                        (original_data[COL_HOME_TEAM] == home_team) &
                        (original_data[COL_AWAY_TEAM] == away_team)
                        ]

                    if original_row.empty:
                        # If we cannot find original metadata, skip this match (should be rare)
                        logging.warning(f"Original metadata not found for {home_team} vs {away_team} (round {round_num})")
                        continue

                    original_row = original_row.iloc[0]

                    # Find most common result from iterations
                    result_counts = Counter(match_data['results'])
                    most_common_result = result_counts.most_common(1)[0][0]

                    # Create final prediction row (Predicted = True)
                    prediction_row = {
                        COL_MATCH_NUMBER: match_data['match_info']['match_number'],
                        COL_ROUND_NUMBER: round_num,
                        COL_DATE: original_row[COL_DATE] if COL_DATE in original_row else '',
                        COL_LOCATION: original_row[COL_LOCATION] if COL_LOCATION in original_row else '',
                        COL_HOME_TEAM: home_team,
                        COL_AWAY_TEAM: away_team,
                        COL_RESULT: most_common_result,
                        COL_PREDICTED: True
                    }
                    final_predictions.append(prediction_row)

                    # If the original data contains an actual result, mark it to be appended later
                    if COL_RESULT in original_row and pd.notna(original_row[COL_RESULT]) and original_row[COL_RESULT] != '':
                        appended_actual_keys.add((home_team, away_team))

                # Next, include ALL actual results present in the original data for this round
                # This covers matches that may not have been predicted (no predictions in memory)
                try:
                    round_original_rows = original_data[original_data[COL_ROUND_NUMBER] == round_number]
                except Exception:
                    round_original_rows = pd.DataFrame()

                if not round_original_rows.empty:
                    actual_rows = round_original_rows[round_original_rows[COL_RESULT].notna() & (round_original_rows[COL_RESULT] != '')]

                    for _, orig in actual_rows.iterrows():
                        home = orig[COL_HOME_TEAM]
                        away = orig[COL_AWAY_TEAM]
                        match_num = orig[COL_MATCH_NUMBER] if COL_MATCH_NUMBER in orig else None

                        key = (home, away)

                        # If we already appended an actual row for this match, skip to avoid duplicates
                        if key in appended_actual_keys:
                            # But ensure we still append the actual row even if we appended the key above
                            # only append once: since appended_actual_keys was used as marker, avoid double-appending
                            continue

                        actual_row = {
                            COL_MATCH_NUMBER: match_num,
                            COL_ROUND_NUMBER: round_number,
                            COL_DATE: orig[COL_DATE] if COL_DATE in orig else '',
                            COL_LOCATION: orig[COL_LOCATION] if COL_LOCATION in orig else '',
                            COL_HOME_TEAM: home,
                            COL_AWAY_TEAM: away,
                            COL_RESULT: orig[COL_RESULT],
                            COL_PREDICTED: False
                        }
                        final_predictions.append(actual_row)
                        appended_actual_keys.add(key)

                # Save final aggregated predictions
                logging.info(f"  Saving to {FINAL_PREDICTIONS_FILE}...")
                final_df = pd.DataFrame(final_predictions)
                # Sort so that predicted rows appear before actual rows for the same match
                final_df = final_df.sort_values([COL_MATCH_NUMBER, COL_PREDICTED], ascending=[True, False]).reset_index(drop=True)
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
    """Main function to run the football predictions.

    The ConfigurationManager may return either a single PredictionConfig or a list
    of PredictionConfig objects (for processing multiple leagues). Normalize to a
    list and run the engine for each config separately.
    """
    from backend.utils.config import ConfigurationManager

    # Set up configuration. ConfigurationManager may return a single PredictionConfig or a list of them
    configs = ConfigurationManager.setup_configuration()

    # Normalize to a list
    if not isinstance(configs, list):
        configs = [configs]

    for cfg in configs:
        logging.info(f"Running predictions for league={cfg.league}, year={cfg.year} rounds={cfg.rounds_to_predict}")
        engine = PredictionEngine(cfg)
        engine.run_predictions()

        # Print summary
        summary = engine.get_predictions_summary()
        logging.info(f"Predictions summary for {cfg.league}: {summary}")

        # Save final aggregated predictions directly from memory (no individual iteration files)
        engine.save_aggregated_predictions_from_memory()

        logging.info(f"Completed processing league={cfg.league}")

    logging.info("All configured leagues completed and final aggregated results saved.")
