from flask import Flask, render_template, request
import pickle
from models.linear_regression import train_model, predict_calories
from models.logistic_Regression import train_logistic, predict_watch

app = Flask(__name__)

# Load model
Linear_model = train_model()
Logistic_model = train_logistic()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/use_cases")
def use_cases():
    return render_template("use_cases.html")

@app.route("/use_cases_alien")
def alien():
    return render_template("alien.html")

# Linear Regression 
@app.route("/form", methods=["GET", "POST"])
def form():
    result = None

    if request.method == "POST":
        duration = float(request.form["duration"])

        result = predict_calories(Linear_model, duration)

    return render_template("form.html", result=result)


@app.route("/use_cases_netflix")
def netflix():
    return render_template("netflix.html")

@app.route("/fraud")
def fraud():
    return render_template("fraud.html")

@app.route("/medical")
def medical():
    return render_template("medical.html")

@app.route("/sales")
def sales():
    return render_template("sales.html")

@app.route("/watch", methods=["GET", "POST"])
def watch():
    result = None
    probability = None

    if request.method == "POST":
        price = float(request.form["price"])
        reviews = float(request.form["reviews"])
        brand = request.form["brand"]

        result, probability = predict_watch(price, reviews, brand)

    return render_template("watch.html", result=result, probability=probability)






if __name__ == "__main__":
    app.run(debug=True)


