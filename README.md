_# FootballPredictor

## Overview

FootballPredictor is a small, experimental project that uses historical football (soccer) results to generate likely
outcomes for upcoming matches. It analyses past games, learns simple patterns, and produces predictions such as likely
scorelines or match outcomes (home win, draw, away win).

---

## Who this is for

- Non-technical users who want a simple, data-driven view of match outcomes.
- Developers, data enthusiasts, or students who want to explore or extend sports-prediction code.

---

## How it works (high-level)

1. The system reads historical match results from CSV files.
2. It computes basic team statistics (form and strength) from past matches.
3. A machine learning model is trained on those features to learn typical result patterns.
4. The model simulates many predictions for upcoming matches and aggregates them into a final recommendation (scoreline
   or outcome).

Note: This is a simple, explainable approach — not a complex neural network — so results are easy to inspect and reason
about.

---

## Where the data comes from

- Input data is CSV (spreadsheet-style) files placed under `backend/files/input/<league>/<year>.csv` (for example:
  `backend/files/input/epl/2025.csv`).
- The project includes a small downloader utility that can fetch CSVs from a public source when available.

---

## What the predictions mean

- Predictions are probabilistic best-effort estimates based on historical data. They are not guarantees.
- The output is either a likely scoreline (for example, `2 - 1`) or a categorical result (home/draw/away).
- The system cannot automatically detect real-world changes (injuries, weather, late transfers) unless that information
  is added to the data.

---

## Where to find results

Final aggregated predictions are written to:

```
backend/files/output/<league>/<year>/<round>/final_predictions.csv
```

By default, the `backend/files/output/` folder is ignored in git (so outputs are not committed). If you want to keep
outputs under version control, remove that path from `.gitignore`.

---