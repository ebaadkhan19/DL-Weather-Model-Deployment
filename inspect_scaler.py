import joblib
import os
import sklearn

print(f"sklearn version: {sklearn.__version__}")

scaler_path = os.path.join("scalers", "scaler_x_Karachi")
try:
    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from {scaler_path}")
    print(f"Expected features: {scaler.n_features_in_}")
    if hasattr(scaler, "feature_names_in_"):
        print("Feature names found:")
        for i, name in enumerate(scaler.feature_names_in_):
            print(f"{i}: {name}")
    else:
        print("No feature names stored in scaler.")
except Exception as e:
    print(f"Error loading scaler: {e}")
