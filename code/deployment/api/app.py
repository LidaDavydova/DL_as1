# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained Wine model (with 5 features)
model = joblib.load("/models/model.pkl")
    
# Define the FastAPI app
app = FastAPI()

# Define the input data schema for Wine features
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    color_intensity: float
    hue: float
    proline: float

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: WineInput):
    # Convert input data to the correct format for the model
    data = np.array([[
        input_data.alcohol,
        input_data.malic_acid,
        input_data.color_intensity,
        input_data.hue,
        input_data.proline
    ]])
    
    prediction = model.predict(data)
    
    # Return the predicted class
    return {"prediction": int(prediction[0])}
