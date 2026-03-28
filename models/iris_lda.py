import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_model():
    data = pd.read_csv("data/iris.csv")
    X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    y = data["Species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train_scaled, y_train)

    accuracy = model.score(X_test_scaled, y_test) 

    return model, scaler, accuracy

def predict_species(model, scaler, values):
    X = scaler.transform([values])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    classes = model.classes_
    prob_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probability)}
    return prediction, prob_dict