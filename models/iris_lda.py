import pandas as pd
import matplotlib.pyplot as pt
import io
import base64
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    pt.figure()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    pt.xlabel("Predicted")
    pt.ylabel("Actual")
    pt.title("Confusion Matrix")

    img = io.BytesIO()
    pt.savefig(img, format='png')
    img.seek(0)

    graph = base64.b64encode(img.getvalue()).decode()
    pt.close()

    return graph

def plot_lda_projection(model,X,y):
    X_lda = model.transform(X)

    pt.figure()

    for label in set(y):
        pt.scatter(
            X_lda[y == label, 0],
            [0]*len(X_lda[y == label]),
            label = label
        )
    pt.xlabel("LDA componet 1")
    pt.title("LDA projection")
    pt.legend()

    img = io.BytesIO()
    pt.savefig(img,format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    pt.close()

    return graph_url

def plot_roc_curve(model, X_test, y_test):

    classes = model.classes_
    y_test_bin = label_binarize(y_test, classes=classes)

    y_score = model.predict_proba(X_test)

    pt.figure()

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        pt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

    pt.plot([0, 1], [0, 1], 'k--')  # línea diagonal
    pt.xlabel("False Positive Rate")
    pt.ylabel("True Positive Rate")
    pt.title("ROC Curve (Multiclass)")
    pt.legend()

    import io, base64
    img = io.BytesIO()
    pt.savefig(img, format='png')
    img.seek(0)
    graph = base64.b64encode(img.getvalue()).decode()
    pt.close()
    return graph


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

    graph = plot_lda_projection(model, X_train_scaled, y_train)
    cm_graph = plot_confusion_matrix(model, X_test_scaled, y_test)

    y_pred = model.predict(X_test_scaled)

    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    roc_graph = plot_roc_curve(model, X_test_scaled, y_test)

    return model, scaler, accuracy, graph, cm_graph, precision, recall, f1, roc_graph

def predict_species(model, scaler, values):
    X = scaler.transform([values])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    classes = model.classes_
    prob_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probability)}
    return prediction, prob_dict