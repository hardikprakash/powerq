import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_zone2 = joblib.load(os.path.join(base_dir, "server", "models", "best_xgb_zone2.pkl"))
model_zone3 = joblib.load(os.path.join(base_dir, "server", "models", "best_xgb_zone3.pkl"))

app = FastAPI(title="Power Consumption Prediction API")

# Request schema
class PredictionRequest(BaseModel):
    Temperature: float
    Humidity: float
    WindSpeed: float
    GeneralDiffuseFlows: float
    PowerConsumption_Zone1: float

@app.post("/predict/zone2")
async def predict_zone2(data: PredictionRequest):
    features = pd.DataFrame([[
        data.Temperature,
        data.Humidity,
        data.WindSpeed,
        data.GeneralDiffuseFlows,
        data.PowerConsumption_Zone1
    ]], columns=[
        "Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "PowerConsumption_Zone1"
    ])
    prediction = model_zone2.predict(features)
    return {"PowerConsumption_Zone2": prediction[0]}

@app.post("/predict/zone3")
async def predict_zone3(data: PredictionRequest):
    features = pd.DataFrame([[
        data.Temperature,
        data.Humidity,
        data.WindSpeed,
        data.GeneralDiffuseFlows,
        data.PowerConsumption_Zone1
    ]], columns=[
        "Temperature", "Humidity", "WindSpeed", "GeneralDiffuseFlows", "PowerConsumption_Zone1"
    ])
    prediction = model_zone3.predict(features)
    return {"PowerConsumption_Zone3": prediction[0]}

@app.get("/")
async def root():
    return {"message": "Power Consumption Prediction API is running!"}
