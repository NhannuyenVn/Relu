from fastapi import FastAPI
import pandas as pd
from sklearn.cluster import KMeans
import joblib

app = FastAPI()
model = joblib.load('kmeans_model.pkl')

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    label = model.predict(df)
    return {"cluster": int(label[0])}