"""
Flask API for serving football predictions.

This API loads prediction CSV files from the output directory and serves them
via REST endpoints for the frontend to consume.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Constants
FINAL_PREDICTIONS_FILE = 'final_predictions.csv'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory cache for predictions
predictions_cache: Dict[str, Dict[str, List[Dict]]] = {}


class PredictionLoader:
    """Handles loading and caching of prediction data."""

    def __init__(self, base_output_dir: str = 'files/output'):
        self.base_output_dir = Path(__file__).parent / base_output_dir
        logger.info(f"Prediction loader initialized with base directory: {self.base_output_dir}")

    def discover_predictions(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Discover all available predictions in the output directory.

        Returns:
            Dict with structure: {league: {year: [rounds]}}
        """
        discoveries = {}

        if not self.base_output_dir.exists():
            logger.warning(f"Output directory does not exist: {self.base_output_dir}")
            return discoveries

        # Iterate through leagues
        for league_dir in self.base_output_dir.iterdir():
            if not league_dir.is_dir():
                continue

            league = league_dir.name
            discoveries[league] = {}

            # Iterate through years
            for year_dir in league_dir.iterdir():
                if not year_dir.is_dir():
                    continue

                year = year_dir.name
                rounds = []

                # Iterate through rounds
                for round_dir in year_dir.iterdir():
                    if not round_dir.is_dir():
                        continue

                    # Check if final_predictions.csv exists
                    predictions_file = round_dir / FINAL_PREDICTIONS_FILE
                    if predictions_file.exists():
                        try:
                            round_num = int(round_dir.name)
                            rounds.append(round_num)
                        except ValueError:
                            logger.warning(f"Invalid round directory name: {round_dir.name}")

                if rounds:
                    discoveries[league][year] = sorted(rounds)

        logger.info(f"Discovered predictions: {discoveries}")
        return discoveries

    def load_predictions(self, league: str, year: str, round_num: int) -> Optional[List[Dict]]:
        """
        Load predictions from CSV file.

        Args:
            league: League name (e.g., 'epl')
            year: Year (e.g., '2025')
            round_num: Round number

        Returns:
            List of prediction dictionaries or None if not found
        """
        predictions_file = (
                self.base_output_dir / league / year / str(round_num) / FINAL_PREDICTIONS_FILE
        )

        if not predictions_file.exists():
            logger.warning(f"Predictions file not found: {predictions_file}")
            return None

        try:
            df = pd.read_csv(predictions_file)
            # Convert DataFrame to list of dictionaries
            predictions = df.to_dict('records')
            logger.info(f"Loaded {len(predictions)} predictions from {predictions_file}")
            return predictions
        except Exception as e:
            logger.error(f"Error loading predictions from {predictions_file}: {e}")
            return None

    def load_all_predictions(self) -> None:
        """Load all available predictions into memory cache."""
        global predictions_cache

        logger.info("Loading all predictions into memory...")
        discoveries = self.discover_predictions()

        total_loaded = 0
        for league, years in discoveries.items():
            if league not in predictions_cache:
                predictions_cache[league] = {}

            for year, rounds in years.items():
                if year not in predictions_cache[league]:
                    predictions_cache[league][year] = {}

                for round_num in rounds:
                    predictions = self.load_predictions(league, year, round_num)
                    if predictions:
                        cache_key = str(round_num)
                        predictions_cache[league][year][cache_key] = predictions
                        total_loaded += len(predictions)

        logger.info(f"Loaded {total_loaded} total predictions into memory")


# Initialize loader
loader = PredictionLoader()


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Football Predictions API'
    })


@app.route('/api/leagues', methods=['GET'])
def get_leagues():
    """Get all available leagues."""
    leagues = list(predictions_cache.keys())
    return jsonify({
        'leagues': leagues
    })


@app.route('/api/leagues/<league>/years', methods=['GET'])
def get_years(league: str):
    """Get all available years for a league."""
    if league not in predictions_cache:
        return jsonify({'error': f'League not found: {league}'}), 404

    years = list(predictions_cache[league].keys())
    return jsonify({
        'league': league,
        'years': years
    })


@app.route('/api/leagues/<league>/years/<year>/rounds', methods=['GET'])
def get_rounds(league: str, year: str):
    """Get all available rounds for a league and year."""
    if league not in predictions_cache:
        return jsonify({'error': f'League not found: {league}'}), 404

    if year not in predictions_cache[league]:
        return jsonify({'error': f'Year not found: {year}'}), 404

    rounds = [int(r) for r in predictions_cache[league][year].keys()]
    rounds.sort()

    return jsonify({
        'league': league,
        'year': year,
        'rounds': rounds
    })


@app.route('/api/predictions/<league>/<year>/<int:round_num>', methods=['GET'])
def get_predictions(league: str, year: str, round_num: int):
    """
    Get predictions for a specific league, year, and round.

    Query parameters:
        - team: Filter by team name (home or away)
    """
    if league not in predictions_cache:
        return jsonify({'error': f'League not found: {league}'}), 404

    if year not in predictions_cache[league]:
        return jsonify({'error': f'Year not found: {year}'}), 404

    round_key = str(round_num)
    if round_key not in predictions_cache[league][year]:
        return jsonify({'error': f'Round not found: {round_num}'}), 404

    predictions = predictions_cache[league][year][round_key]

    # Optional filtering by team
    team_filter = request.args.get('team')
    if team_filter:
        predictions = [
            p for p in predictions
            if team_filter.lower() in p.get('Home Team', '').lower() or
               team_filter.lower() in p.get('Away Team', '').lower()
        ]

    return jsonify({
        'league': league,
        'year': year,
        'round': round_num,
        'count': len(predictions),
        'predictions': predictions
    })


@app.route('/api/predictions', methods=['GET'])
def get_all_available_predictions():
    """Get summary of all available predictions."""
    summary = {}

    for league, years in predictions_cache.items():
        summary[league] = {}
        for year, rounds in years.items():
            summary[league][year] = {
                'rounds': sorted([int(r) for r in rounds.keys()]),
                'total_predictions': sum(len(predictions) for predictions in rounds.values())
            }

    return jsonify(summary)


@app.route('/api/reload', methods=['POST'])
def reload_predictions():
    """Reload all predictions from disk into memory."""
    try:
        loader.load_all_predictions()
        return jsonify({
            'status': 'success',
            'message': 'Predictions reloaded successfully'
        })
    except Exception as e:
        logger.error(f"Error reloading predictions: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def init_api():
    """Initialize the API by loading all predictions."""
    logger.info("Initializing Football Predictions API...")
    loader.load_all_predictions()
    logger.info("API initialization complete")


if __name__ == '__main__':
    # Load predictions on startup
    init_api()

    # Run the Flask app
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'

    logger.info(f"Starting Flask API on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
