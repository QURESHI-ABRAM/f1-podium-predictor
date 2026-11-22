# F1 Podium Predictor

A machine learning classification project built with Python, pandas, and scikit-learn to predict Formula 1 top-3 podium finishes using pre-race driver and constructor form from historical Kaggle race data.

## Project Overview
- **Task:** Predict top-3 race finishers (podium) using strictly pre-race features so no race-day outcomes or post-race data leak into training or inference.
- **Dataset:** Historical Formula 1 CSV tables (results.csv, races.csv, and related tables from archive.zip) from Kaggle (1950-2024)[cite: 1].
- **Evaluation Split:** Chronological train/test split—trained on the 2000-2021 seasons and evaluated on 2022-2024 (1,359 test entries), ranking the top 3 predicted podium probabilities per race.

## Pre-Race Features
- **Starting Grid Position (grid):** Qualifying/grid slot with pit-lane or unclassified 0 values mapped to 20.
- **Driver Form:** Cumulative season points entering the race (driver_points_before), rolling 5-race average finish position (driver_last5_finish), and rolling 5-race podium rate (driver_last5_podiums), all shifted by one race to prevent leakage.
- **Constructor Form:** Cumulative season points entering the race (team_points_before) and rolling 5-race podium rate (team_last5_podiums) aggregated at the constructor-race level.

## Results (2022-2024 Holdout — 1,359 Rows)
| Model / Baseline | Precision | Accuracy | Notes |
| :--- | :---: | :---: | :--- |
| Grid Top-3 Baseline | 65.0% | 89.5% | Predicts top 3 starting grid slots |
| Initial Random Forest (1950-2021 train) | 61.3% | 88.1% | Trained across full historical span without team form |
| Gradient Boosting Top-3 (2000-2021 train) | 65.2% | 89.6% | Pre-race driver + constructor rolling features, top-3 per race |

## Visualizations
- podium_by_grid.png — Historical Formula 1 podium conversion rate across starting grid positions 1-10.
- feature_importance.png — Relative importance of pre-race features in the Gradient Boosting classifier.

## Setup & Usage
~~~bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_podium_model.py
~~~
