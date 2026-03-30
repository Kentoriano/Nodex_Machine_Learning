import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

    X = pd.get_dummies(data[["price", "number_of_reviews", "brand_name"]], columns=["brand_name"])
    y = data["target"]

    scaler = StandardScaler()
    X[["price", "number_of_reviews"]] = scaler.fit_transform(X[["price", "number_of_reviews"]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm       = confusion_matrix(y_test, y_pred).tolist()
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred) * 100, 2)
    recall   = round(recall_score(y_test, y_pred) * 100, 2)
    f1       = round(f1_score(y_test, y_pred) * 100, 2)

    return model, cm, accuracy, precision, recall, f1

def predict_watch(price, reviews, brand_name):
    global model

    input_data = {
        "price": price,
        "number_of_reviews": reviews
    }

    model_columns = model.feature_names_in_

    for col in model_columns:
        if "brand_name_" in col:
            input_data[col] = 0
        
    brand_col = f"brand_name_{brand_name}"
    if brand_col in input_data:
        input_data[brand_col] = 1

    input_df = pd.DataFrame([input_data])
    input_df = pd.DataFrame([input_data])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability

