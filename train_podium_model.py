from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

DATA_DIR = Path("data")

results = pd.read_csv(DATA_DIR / "results.csv")
races = pd.read_csv(DATA_DIR / "races.csv")[["raceId", "year", "round", "circuitId", "name"]]

df = results.merge(races, on="raceId", how="inner")

df["grid"] = pd.to_numeric(df["grid"], errors="coerce").replace(0, 20).fillna(20)
df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
df["podium"] = (df["positionOrder"] <= 3).astype(int)

df = df.sort_values(["year", "round", "driverId"]).reset_index(drop=True)

df["driver_points_before"] = df.groupby(["year", "driverId"])["points"].cumsum() - df["points"]
df["driver_last5_finish"] = (
    df.groupby("driverId")["positionOrder"]
    .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    .fillna(12.0)
)
df["driver_last5_podiums"] = (
    df.groupby("driverId")["podium"]
    .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    .fillna(0.0)
)

team_race = (
    df.groupby(["year", "round", "constructorId"], as_index=False)
    .agg(team_points=("points", "sum"), team_podiums=("podium", "sum"))
    .sort_values(["year", "round", "constructorId"])
)
team_race["team_points_before"] = (
    team_race.groupby(["year", "constructorId"])["team_points"].cumsum() - team_race["team_points"]
)
team_race["team_last5_podiums"] = (
    team_race.groupby("constructorId")["team_podiums"]
    .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    .fillna(0.0)
)

df = df.merge(
    team_race[["year", "round", "constructorId", "team_points_before", "team_last5_podiums"]],
    on=["year", "round", "constructorId"],
    how="left",
)

features = [
    "grid",
    "driver_points_before",
    "driver_last5_finish",
    "driver_last5_podiums",
    "team_points_before",
    "team_last5_podiums",
]
target = "podium"

train_df = df[(df["year"] >= 2000) & (df["year"] <= 2021)].copy()
test_df = df[(df["year"] >= 2022) & (df["year"] <= 2024)].copy()

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

baseline_pred = ((test_df["grid"] >= 1) & (test_df["grid"] <= 3)).astype(int)
baseline_prec = precision_score(y_test, baseline_pred, zero_division=0)
baseline_acc = accuracy_score(y_test, baseline_pred)

model = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
)
model.fit(X_train, y_train)

test_df["podium_prob"] = model.predict_proba(X_test)[:, 1]
test_df["pred_rank"] = test_df.groupby("raceId")["podium_prob"].rank(ascending=False, method="first")
pred = (test_df["pred_rank"] <= 3).astype(int)

model_prec = precision_score(y_test, pred, zero_division=0)
model_acc = accuracy_score(y_test, pred)

print(f"Test rows (2022-2024): {len(test_df)}")
print(f"Grid Top-3 Baseline      -> Precision: {baseline_prec:.1%} | Accuracy: {baseline_acc:.1%}")
print(f"Gradient Boosting Top-3  -> Precision: {model_prec:.1%} | Accuracy: {model_acc:.1%}")

podium_by_grid = df[df["grid"].between(1, 10)].groupby("grid")["podium"].mean() * 100
plt.figure(figsize=(8, 4.5), dpi=150)
podium_by_grid.plot(kind="bar", color="#e10600")
plt.title("Historical F1 Podium Rate by Starting Grid Position (1-10)")
plt.xlabel("Grid Slot")
plt.ylabel("Podium Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("podium_by_grid.png")

importances = pd.Series(model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(8, 4.5), dpi=150)
importances.plot(kind="barh", color="#1f2937")
plt.title("Pre-Race Feature Importance for F1 Podium Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved charts: podium_by_grid.png and feature_importance.png")
