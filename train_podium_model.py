from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

DATA_DIR = Path("data")

TEAM_COLORS = {
    "McLaren": "\033[38;5;208m",      # Papaya orange
    "Mercedes": "\033[38;5;44m",      # Teal / Petronas cyan
    "Red Bull": "\033[38;5;33m",      # Deep blue
    "Ferrari": "\033[38;5;196m",      # Rosso Corsa red
    "Aston Martin": "\033[38;5;29m",  # British racing green
    "Alpine F1 Team": "\033[38;5;198m",
    "Williams": "\033[38;5;39m",
    "RB F1 Team": "\033[38;5;63m",
    "Sauber": "\033[38;5;46m",
    "Haas F1 Team": "\033[38;5;245m",
}
DRIVER_PALETTE = [
    "\033[96m",  # Bright cyan
    "\033[93m",  # Bright yellow
    "\033[92m",  # Bright green
    "\033[95m",  # Bright magenta
    "\033[94m",  # Bright blue
    "\033[91m",  # Bright red
]
RESET = "\033[0m"
BOLD = "\033[1m"


def colorize_driver(name: str) -> str:
    idx = sum(ord(ch) for ch in name) % len(DRIVER_PALETTE)
    return f"{BOLD}{DRIVER_PALETTE[idx]}{name}{RESET}"


def colorize_team(team: str) -> str:
    for key, code in TEAM_COLORS.items():
        if key.lower() in team.lower():
            return f"{BOLD}{code}{team}{RESET}"
    idx = sum(ord(ch) for ch in team) % len(DRIVER_PALETTE)
    return f"{DRIVER_PALETTE[idx]}{team}{RESET}"


results = pd.read_csv(DATA_DIR / "results.csv")
races = pd.read_csv(DATA_DIR / "races.csv")[["raceId", "year", "round", "circuitId", "name"]]
drivers = pd.read_csv(DATA_DIR / "drivers.csv")[["driverId", "forename", "surname"]]
constructors = pd.read_csv(DATA_DIR / "constructors.csv")[["constructorId", "name"]].rename(
    columns={"name": "team_name"}
)

drivers["driver_name"] = (
    drivers["forename"].astype(str).str.strip() + " " + drivers["surname"].astype(str).str.strip()
)

df = results.merge(races, on="raceId", how="inner")
df = df.merge(drivers[["driverId", "driver_name"]], on="driverId", how="left")
df = df.merge(constructors, on="constructorId", how="left")

df["driver_name"] = df["driver_name"].fillna("Driver " + df["driverId"].astype(str))
df["team_name"] = df["team_name"].fillna("Team " + df["constructorId"].astype(str))

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

sample_races = test_df[["year", "round", "name", "raceId"]].drop_duplicates().tail(3)
print("\nSample Top-3 Predicted Podiums (Color-Coded Driver & Team Names):")
for _, race in sample_races.iterrows():
    race_preds = (
        test_df[test_df["raceId"] == race["raceId"]]
        .sort_values("pred_rank")
        .head(3)
    )
    print(f"\n{BOLD}{race['year']} R{race['round']} - {race['name']}{RESET}:")
    for _, row in race_preds.iterrows():
        actual_pos = int(row["positionOrder"]) if pd.notna(row["positionOrder"]) else -1
        d_colored = colorize_driver(row["driver_name"])
        t_colored = colorize_team(row["team_name"])
        print(
            f"  #{int(row['pred_rank'])} {d_colored} ({t_colored}) "
            f"| prob={row['podium_prob']:.1%} | grid={int(row['grid'])} | actual_finish=P{actual_pos}"
        )

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
print("\nSaved charts: podium_by_grid.png and feature_importance.png")
