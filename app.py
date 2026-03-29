import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request
import pickle
from models.linear_regression import train_model, predict_calories
from models.logistic_Regression import train_logistic, predict_watch

app = Flask(__name__)

Linear_model = train_model()
Logistic_model, cm, accuracy, precision, recall, f1 = train_logistic()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/use_cases")
def use_cases():
    return render_template("use_cases.html")

@app.route("/use_cases_alien")
def alien():
    return render_template("alien.html")
 
@app.route("/form", methods=["GET", "POST"])
def form():
    result = None
    show_graph = False

    if request.method == "POST":
        duration = float(request.form["duration"])

        result = predict_calories(Linear_model, duration)

        x = np.linspace(0, 60, 100)
        y = Linear_model.predict(x.reshape(-1, 1))

        plt.figure()
        plt.plot(x, y)  # regression line
        plt.scatter(duration, result)  # user point

        plt.xlabel("Duration")
        plt.ylabel("Calories")
        plt.title("Linear Regression Prediction")

        plt.savefig("static/img/graph.png")
        plt.close()

        show_graph = True

    return render_template("form.html", result=result, show_graph=show_graph)


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
        probability = round(float(probability) * 100, 2)

    return render_template("watch.html", result=result, probability=probability,
                           cm=cm, accuracy=accuracy,
                           precision=precision, recall=recall,f1=f1)


@app.route('/linear-regression-concepts')
def linear_regression_concepts():
    return render_template('linear_regression_concepts.html')



if __name__ == "__main__":
    app.run(debug=True)