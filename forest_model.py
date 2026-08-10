import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the CSV
df = pd.read_csv("data/training_sensor_data.csv")

# 2. Inspect the dataset
print(df.head())
print(df.columns)
print(df["actual_condition"].value_counts())

# 3. Clean column names and text labels
df.columns = df.columns.str.strip()

df["actual_condition"] = (
    df["actual_condition"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Optional: exclude manual_stop because it is a command event,
# not a sensor-detectable mechanical fault.
df = df[df["actual_condition"].isin([
    "normal",
    "obstruction",
    "overload",
])].copy()

# 4. Choose only real sensor-based features
FEATURES = [
    "position_cm",
    "speed_cm_s",
    "motor_current_a",
    "sensor_distance_cm",
    "weight_kg",
]

TARGET = "actual_condition"

# 5. Ensure sensor columns are numeric and remove invalid rows
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=FEATURES + [TARGET])

# 6. Create ML dataset
X = df[FEATURES]
y = df[TARGET]

print("\nFeature dataset shape:", X.shape)
print("Target labels:", y.unique())

# 7. Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 8. Train the Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 9. Evaluate it
predictions = model.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["normal", "obstruction", "overload"],
))

# 10. Check which sensors most influenced predictions
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)
joblib.dump(model, "random_forest_model/random_forest.joblib")