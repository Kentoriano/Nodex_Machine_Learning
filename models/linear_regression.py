import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model():
    data = pd.read_csv("data/train.csv")

    X = data[["Duration"]]   # independent variable
    y = data["Calories"]     # dependent variable

    model = LinearRegression()
    model.fit(X, y)

    return model

def predict_calories(model, value):
    prediction = model.predict([[value]])
    return prediction[0]