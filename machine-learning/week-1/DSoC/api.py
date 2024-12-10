from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import sklearn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
# Declaring our FastAPI instance
app = FastAPI()

#importing downloaded model, scaler and encoder
loaded_model = joblib.load('rf_classifier.joblib')
scaler = joblib.load('zscore_scaler.joblib')
enc = joblib.load('encoder.joblib')


class request_body(BaseModel):
    TransactionAmt : float
    P_emaildomain : str
    id_31 : str                         #browser
    card4 : str                         #card company
    card6 : str                         #card type
    DeviceType : str

#data preprocessing pipeline
def processData(data: request_body) -> pd.DataFrame:
    # scaling TransactionAmt
    scaled_amt = scaler.transform([[data.TransactionAmt]])[0, 0]

    # combining P_emaildomain and id_31 for encoding
    sparse_features_input = [data.P_emaildomain, data.id_31]
    sparse_features = enc.transform([sparse_features_input]).toarray()

    # Binary features for card4, card6, and DeviceType
    card4_features = {
        "card4_american express": 0,
        "card4_discover": 0,
        "card4_mastercard": 0,
        "card4_visa": 0
    }
    card6_features = {
        "card6_charge card": 0,
        "card6_credit": 0,
        "card6_debit": 0
    }
    device_features = {
        "DeviceType_desktop": 0,
        "DeviceType_mobile": 0
    }

    # setting the appropriate binary feature to 1
    if data.card4 in card4_features:
        card4_features[f"card4_{data.card4}"] = 1
    if data.card6 in card6_features:
        card6_features[f"card6_{data.card6}"] = 1
    if data.DeviceType in device_features:
        device_features[f"DeviceType_{data.DeviceType}"] = 1

    # combining all features into a single DataFrame
    combined_features = [scaled_amt] + sparse_features.flatten().tolist()
    combined_features += list(card4_features.values()) + list(card6_features.values()) + list(device_features.values())

    feature_names = (
        ["TransactionAmt"] +
        list(card4_features.keys()) +
        list(card6_features.keys()) +
        list(device_features.keys()) +
        [str(i) for i in range(sparse_features.shape[1])] 
        
    )

    k = pd.DataFrame([combined_features], columns=feature_names)
    return k

#adding an endpoint 
@app.post('/predict')
def predict(preData : request_body):
    data = processData(preData)
    
    class_idx = loaded_model.predict(data)[0]
    return { 'class' : 'Fraud ' if class_idx == 1 else 'Not Fraud'}
