import pandas as pd

df = pd.read_csv("data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

df_encoded = pd.get_dummies(df)

row = df_encoded.iloc[0].drop("Churn_Yes")  # remove target column

# Print the list of 46 encoded features
print(row.tolist())
