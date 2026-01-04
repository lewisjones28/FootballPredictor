import React, { useState, useEffect } from 'react';
import './App.css';
import PredictionCard from './components/PredictionCard';
import type { Prediction } from './types/index.ts';
import { loadPredictions, getAvailableLeagues, getAvailableRounds } from './utils/dataLoader';

function App() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLeague, setSelectedLeague] = useState<string>('epl');
  const [selectedRound, setSelectedRound] = useState<string>('20');
  const [availableLeagues, setAvailableLeagues] = useState<string[]>(['epl']);
  const [availableRounds, setAvailableRounds] = useState<string[]>(['20', '21']);
  const [filterType, setFilterType] = useState<'all' | 'predictions' | 'results'>('all');

  // Load available leagues on mount
  useEffect(() => {
    const fetchLeagues = async () => {
      try {
        const leagues = await getAvailableLeagues();
        setAvailableLeagues(leagues);
        if (leagues.length > 0 && !leagues.includes(selectedLeague)) {
          setSelectedLeague(leagues[0]);
        }
      } catch (err) {
        console.error('Error fetching leagues:', err);
      }
    };
    fetchLeagues();
  }, []);

  // Load available rounds when league changes
  useEffect(() => {
    const fetchRounds = async () => {
      try {
        const rounds = await getAvailableRounds(selectedLeague);
        setAvailableRounds(rounds);
        if (rounds.length > 0 && !rounds.includes(selectedRound)) {
          setSelectedRound(rounds[0]);
        }
      } catch (err) {
        console.error('Error fetching rounds:', err);
      }
    };
    if (selectedLeague) {
      fetchRounds();
    }
  }, [selectedLeague]);

  // Load predictions when league or round changes
  useEffect(() => {
    const fetchPredictions = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await loadPredictions(selectedLeague, selectedRound);
        setPredictions(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load predictions');
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [selectedLeague, selectedRound]);

  // Filter predictions based on filterType
  const filteredPredictions = predictions.filter(prediction => {
    if (filterType === 'all') return true;
    if (filterType === 'predictions') return prediction['Predicted'] === true;
    if (filterType === 'results') return prediction['Predicted'] === false;
    return true;
  });

  return (
    <div className="app">
      <header className="header">
        <div className="header-logo">
          <img src="/football-predictor.svg" alt="Football Predictor Logo" />
        </div>
        <h1>AI Football Match Predictor</h1>
        <p>Powered by thousands of simulations analysing team form, results, and historical data to predict match outcomes</p>
      </header>

      <div className="filters">
        <div className="filter-group">
          <label htmlFor="league-select">League</label>
          <select
            id="league-select"
            value={selectedLeague}
            onChange={(e) => setSelectedLeague(e.target.value)}
          >
            {availableLeagues.map((league) => (
              <option key={league} value={league}>
                {league.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="round-select">Round</label>
          <select
            id="round-select"
            value={selectedRound}
            onChange={(e) => setSelectedRound(e.target.value)}
          >
            {availableRounds.map((round) => (
              <option key={round} value={round}>
                Round {round}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="type-select">Show</label>
          <select
            id="type-select"
            value={filterType}
            onChange={(e) => setFilterType(e.target.value as 'all' | 'predictions' | 'results')}
          >
            <option value="all">All Matches</option>
            <option value="predictions">Predictions Only</option>
            <option value="results">Results Only</option>
          </select>
        </div>
      </div>

      {loading && <div className="loading">Loading predictions...</div>}

      {error && <div className="error">Error: {error}</div>}

      {!loading && !error && filteredPredictions.length === 0 && (
        <div className="no-predictions">
          No {filterType === 'predictions' ? 'predictions' : filterType === 'results' ? 'results' : 'matches'} available for this league and round.
        </div>
      )}

      {!loading && !error && filteredPredictions.length > 0 && (
        <div className="predictions-container">
          {filteredPredictions.map((prediction, index) => (
            <PredictionCard key={index} prediction={prediction} />
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
