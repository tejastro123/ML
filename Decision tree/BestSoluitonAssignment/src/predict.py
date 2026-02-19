
import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    return model.predict_proba(df)[0][1]
