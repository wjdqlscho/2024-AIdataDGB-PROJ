import pymongo
import certifi
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

MONGO_URI = "mongodb+srv:/개인정보=========="
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
collection = client['project_db']['input_data']


def analyze_data():
    data = list(collection.find())

    X = np.array([d['amount'] for d in data]).reshape(-1, 1)
    y = np.array([0 for _ in data]) 

    model = LinearRegression()
    model.fit(X, y)
    print("AI Model trained.")

    results = []
    for d in data:
        prediction = model.predict([[d['amount']]])  
        results.append({
            "name": d['name'],
            "description": d['description'],
            "amount": d['amount'],
            "predicted_risk": float(prediction[0]),  
            "analyzed_at": datetime.now()  
        })
    return results
