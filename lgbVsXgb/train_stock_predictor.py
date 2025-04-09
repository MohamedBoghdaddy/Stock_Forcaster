import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
from math import sqrt

# === Load Data ===
df = pd.read_csv("processed_features.csv", index_col=0)

# === Prepare Features and Labels ===
X = df.drop(columns=['Close'])
y = df['Close']

# === Split Dataset ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# === Normalize ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# === LightGBM Training ===
print("Training LightGBM...")
lgb_train = lgb.Dataset(X_train_scaled, y_train)
lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'verbose': -1
}

lgb_model = lgb.train(
    params=lgb_params,
    train_set=lgb_train,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(5)]
)

lgb_preds = lgb_model.predict(X_test_scaled, num_iteration=lgb_model.best_iteration)
lgb_rmse = sqrt(mean_squared_error(y_test, lgb_preds))
lgb_r2 = r2_score(y_test, lgb_preds)

print(f"[LightGBM] RMSE: {lgb_rmse:.3f}, R²: {lgb_r2:.3f}")

# Save the model
lgb_model.save_model("lgb_model.txt")


# === XGBoost Training ===
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,
    n_estimators=500,
    early_stopping_rounds=5,
    verbosity=0
)

xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

xgb_preds = xgb_model.predict(X_test_scaled)
xgb_rmse = sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)

print(f"[XGBoost] RMSE: {xgb_rmse:.3f}, R²: {xgb_r2:.3f}")

# Save the model
joblib.dump(xgb_model, "xgb_model.pkl")

# === Compare Best Model ===
if xgb_rmse < lgb_rmse:
    print("\n✅ XGBoost performed better. Saved as 'xgb_model.pkl'")
else:
    print("\n✅ LightGBM performed better. Saved as 'lgb_model.txt'")
