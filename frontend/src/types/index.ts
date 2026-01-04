export interface Prediction {
  'Match Number': number;
  'Round Number': number;
  'Date': string;
  'Location': string;
  'Home Team': string;
  'Away Team': string;
  'Result': string; // Format: "2 - 1"
  'Predicted': boolean;
}

export interface PredictionsData {
  league: string;
  round: string;
  predictions: Prediction[];
}

