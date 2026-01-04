import type { Prediction } from '../types/index.ts';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

/**
 * Loads predictions for a given league and round from the API.
 *
 * @param {string} league - The league to load predictions for.
 * @param {string} round - The round to load predictions for.
 * @returns {Promise<Prediction[]>} A promise that resolves to an array of predictions.
 * @throws {Error} If the API request fails.
 */
export const loadPredictions = async (league: string, round: string): Promise<Prediction[]> => {
  try {
    const year = '2025'; // Could be dynamic later
    const response = await fetch(`${API_BASE_URL}/predictions/${league}/${year}/${round}`);

    if (!response.ok) {
      throw new Error(`Failed to load predictions: ${response.statusText}`);
    }

    const data = await response.json();
    return data.predictions || [];
  } catch (error) {
    console.error('Error loading predictions:', error);
    throw error;
  }
};

/**
 * Fetches available league names from the API.
 *
 * @returns {Promise<string[]>} A promise that resolves to an array of league names as strings.
 * @throws {Error} If the API request fails.
 */
export const getAvailableLeagues = async (): Promise<string[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/leagues`);
    if (!response.ok) {
      throw new Error('Failed to fetch leagues');
    }
    const data = await response.json();
    return data.leagues || [];
  } catch (error) {
    console.error('Error fetching leagues:', error);
    // Fallback to static list
    return ['epl'];
  }
};

/**
 * Fetches available rounds numbers for a given league and year.
 *
 * @param {string} league The league to fetch rounds numbers for.
 * @param {string} [year='2025'] The year to fetch round numbers for. Defaults to '2025'.
 * @returns {Promise<string[]>} A promise that resolves to an array of round numbers as strings.
 * @throws {Error} If the API request fails.
 */
export const getAvailableRounds = async (league: string, year: string = '2025'): Promise<string[]> => {
  try {
    const response = await fetch(`${API_BASE_URL}/leagues/${league}/years/${year}/rounds`);
    if (!response.ok) {
      throw new Error('Failed to fetch rounds');
    }
    const data = await response.json();
    return data.rounds.map((r: number) => String(r)) || [];
  } catch (error) {
    console.error('Error fetching rounds:', error);
    // Fallback to static list
    return ['20', '21'];
  }
};

