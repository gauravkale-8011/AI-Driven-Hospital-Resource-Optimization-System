import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------------- LOAD DATA ----------------
df = pd.read_csv("../data/hospital_dataset_pro.csv")

# ---------------- FEATURE ENGINEERING ----------------
df['date'] = pd.to_datetime(df['date'])

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.weekday

# ---------------- SORT (VERY IMPORTANT FOR TIME SERIES) ----------------
df = df.sort_values('date')

# ---------------- TIME SERIES FEATURES ----------------
df['lag_1'] = df['patients'].shift(1)
df['lag_2'] = df['patients'].shift(2)
df['rolling_mean'] = df['patients'].rolling(3).mean()

# Drop NaN rows
df = df.dropna()

# ---------------- FEATURES & TARGET ----------------
X = df.drop(['patients', 'date'], axis=1)
y = df['patients']

# ---------------- TIME-BASED SPLIT (CORRECT WAY) ----------------
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# ---------------- MODEL ----------------
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open("model_patients.pkl", "wb"))

print("✅ Model trained and saved successfully!")