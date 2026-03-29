import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

model = None
encoder = None
cm = None
accuracy = None
precision = None
recall = None
f1 = None

def train_logistic():
    global model, encoder, cm, accuracy, precision, recall, f1

    data = pd.read_csv("data/dataset_watches.csv")
    data["price"] = data["price"].str.replace(",", "")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["number_of_reviews"] = pd.to_numeric(data["number_of_reviews"], errors="coerce")
    data = data.dropna()

    data["target"] = (data["rating"] >= 4.3).astype(int)

    X = data[["price", "number_of_reviews", "brand_name"]].copy()
    y = data["target"]

    encoder = LabelEncoder()
    X["brand_name"] = encoder.fit_transform(X["brand_name"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm       = confusion_matrix(y_test, y_pred).tolist()
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred) * 100, 2)
    recall   = round(recall_score(y_test, y_pred) * 100, 2)
    f1       = round(f1_score(y_test, y_pred) * 100, 2)

    return model, cm, accuracy, precision, recall, f1

def predict_watch(price, reviews, brand_name):
    global model, encoder

    brand_encoded = encoder.transform([brand_name])[0]
    prediction = model.predict([[price, reviews, brand_encoded]])
    probability = model.predict_proba([[price, reviews, brand_encoded]])[0][1]
    return prediction[0], probability