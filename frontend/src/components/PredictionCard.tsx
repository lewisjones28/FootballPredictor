import React from 'react';
import type { Prediction } from '../types';

interface PredictionCardProps {
  prediction: Prediction;
}

/**
 * A card component that displays a prediction for a match.
 *
 * @param {PredictionCardProps} props - The properties for the component.
 * @param {Prediction} props.prediction - The prediction for the match.
 *
 * @returns {React.ReactElement} A React element representing the prediction card.
 */
const PredictionCard: React.FC<PredictionCardProps> = ({ prediction }) => {
  const isPredicted = prediction['Predicted'];


/**
 * Parses a string representing a score and returns an object with home and away scores.
 * If the string is not in the format <number>-<number>, returns null.
 *
 * @example
 * parseScore('2-1') // { home: 2, away: 1 }
 * parseScore('foo-bar') // null
 * @param {string | undefined} result - The string to parse.
 * @returns {{ home: number, away: number } | null} - The parsed score or null if the string is invalid.
 */
  const parseScore = (result: string | undefined): { home: number; away: number } | null => {
    if (!result) return null;
    const match = result.match(/(\d+)\s*-\s*(\d+)/);
    if (match) {
      return { home: parseInt(match[1]), away: parseInt(match[2]) };
    }
    return null;
  };  

  const score = parseScore(prediction['Result']);

/**
 * Returns the type of the result, which can be one of 'H', 'D', or 'A' for home win, draw, or away win respectively.
 * If the score is invalid, returns null.
 */
  const getResultType = (): 'H' | 'D' | 'A' | null => {
    if (!score) return null;
    if (score.home > score.away) return 'H';
    if (score.home < score.away) return 'A';
    return 'D';
  };

  const resultType = getResultType();

/**
 * Returns a class name representing the result of a match.
 *
 * @param {result} result - The type of the result, which can be one of 'H', 'D', or 'A' for home win, draw, or away win respectively.
 * If the result is invalid, returns an empty string.
 *
 * @returns {string} - The class name representing the result of a match.
 */
  const getResultClass = (result?: 'H' | 'D' | 'A' | null): string => {
    if (!result) return '';
    return `result-${result.toLowerCase()}`;
  };

/**
 * Returns the date string as-is from the API.
 *
 * @param {string} dateStr - The date string from the API.
 * @returns {string} - The original date string.
 */
  const formatDate = (dateStr: string): string => {
    return dateStr;
  };

  return (
    <div className={`prediction-card ${isPredicted ? 'is-prediction' : 'is-result'}`}>
      <div className="match-status-badge">
        {isPredicted ? '🔮 Prediction' : '✓ Result'}
      </div>
      <div className={`match-info ${resultType === 'D' ? 'match-info-draw' : ''}`}>
        <div className="teams">
          <div className="team home-team">
            <div className="team-crest">🛡️</div>
            <span className="team-name">{prediction['Home Team']}</span>
          </div>
          <div className="vs">vs</div>
          <div className="team away-team">
            <span className="team-name">{prediction['Away Team']}</span>
            <div className="team-crest">🛡️</div>
          </div>
        </div>
        {score && (
          <div className="predicted-score">
            <span>{score.home}</span>
            <span>-</span>
            <span>{score.away}</span>
          </div>
        )}
        {resultType && (
          <div className={`predicted-result ${getResultClass(resultType)} result-align-${resultType.toLowerCase()}`}>
            {resultType === 'H' ? 'Home Win' :
             resultType === 'D' ? 'Draw' : 'Away Win'}
          </div>
        )}
      </div>
      <div className="match-details">
        {prediction['Date'] && (
          <div className="detail-item">
            <span className="detail-label">🕒</span>
            <span className="detail-value">{formatDate(prediction['Date'])}</span>
          </div>
        )}
        {prediction['Location'] && (
          <div className="detail-item">
            <span className="detail-label">📍</span>
            <span className="detail-value">{prediction['Location']}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionCard;

