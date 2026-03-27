import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

model = None
encoder = None

def train_logistic():
    global model, encoder

    data = pd.read_csv("data/dataset_watches.csv")
    data["price"] = data["price"].str.replace(",", "")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data["number_of_reviews"] = pd.to_numeric(data["number_of_reviews"], errors="coerce")
    data = data.dropna()

    data["target"] = (data["rating"] >=4.3).astype(int)

    X = data[["price", "number_of_reviews", "brand_name"]].copy()
    y = data["target"]

    encoder = LabelEncoder()
    X["brand_name"] = encoder.fit_transform(X["brand_name"])

    model = LogisticRegression(max_iter=1000)
    model.fit(X,y)
    return model

def predict_watch(price, reviews, brand_name):
    global model, encoder

    brand_encoded = encoder.transform([brand_name])[0]
    prediction = model.predict([[price, reviews, brand_encoded]])
    probability = model.predict_proba([[price, reviews, brand_encoded]])[0][1]
    return prediction[0], probability