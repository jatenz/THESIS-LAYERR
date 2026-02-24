# ENHANCED MARKOV CHAIN–BASED VIEWER LOYALTY MANAGEMENT SYSTEM (Algorithm-focused)

This project implements the algorithmic core (no dashboard):
- Offline CSV ingestion (YouTube Analytics export formats)
- Feature engineering and loyalty score
- State classification into loyalty states
- Enhanced Markov Chain with:
  - Laplace smoothing (alpha)
  - Absorbing churn state (optional)
  - Steady-state distribution
  - k-step forecasting via matrix powers
  - Expected time-to-churn (absorbing Markov fundamental matrix)
- Optional HMM + Viterbi decoding of latent states
- Optional Leiden clustering of transition-pattern vectors (requires igraph + leidenalg)
- Evaluation utilities (macro F1 + confusion matrix for next-step prediction)

## Run
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py --csv "Table data.csv" --freq W
```

## Key outputs (written to outputs/)
- time_series_states.csv
- transition_counts.csv
- transition_probabilities.csv
- steady_state.csv
- forecast_kstep.csv
- expected_time_to_churn.csv (if churn enabled)
- evaluation.csv
- confusion_matrix.csv
- summary.csv
- viewer_loyalty.db
