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
