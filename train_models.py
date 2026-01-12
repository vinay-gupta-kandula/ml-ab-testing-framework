import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID, handle missing
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# One-hot encode
df = pd.get_dummies(df)

# Split into X, y
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model A: Logistic Regression
model_a = LogisticRegression(max_iter=500)
model_a.fit(X_train, y_train)

# Model B: RandomForest
model_b = RandomForestClassifier(n_estimators=150, random_state=42)
model_b.fit(X_train, y_train)

# Save models
joblib.dump(model_a, "api/models/model_A.pkl")
joblib.dump(model_b, "api/models/model_B.pkl")

# Save feature columns (IMPORTANT)
joblib.dump(list(X.columns), "api/models/feature_cols.pkl")

# Debug print
print("Models saved successfully!")
print(f"Number of encoded features: {len(X.columns)}")
print("Example feature list:", X.columns.tolist())
