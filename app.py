import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request
from models.linear_regression import train_model, predict_calories
from models.iris_lda import train_model as train_lda_model, predict_species
from models.logistic_Regression import train_logistic, predict_watch

app = Flask(__name__)
linear_model = train_model()
lda_model, scaler, accuracy, graph, cm_graph, precision, recall, f1, roc_graph = train_lda_model()
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


@app.route("/iris", methods=["GET", "POST"])
def ldaf():
    prediction = None
    probabilities = None
    if request.method == "POST":
        sepal_length = float(request.form["sepal_length"])
        sepal_width  = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width  = float(request.form["petal_width"])
        values = [sepal_length, sepal_width, petal_length, petal_width]
        prediction, probabilities = predict_species(lda_model, scaler, values)
        
    return render_template("iris.html", prediction=prediction, probabilities=probabilities, accuracy=round(accuracy * 100,2), graph=graph, cm_graph=cm_graph, roc_graph=roc_graph,
        precision=round(precision, 2), recall = round(recall, 2), f1 = round(f1, 2)
    ) 


@app.route('/lda')
def lda():
    return render_template('linear_discriminant_analysis.html')
  
@app.route('/linear-regression-concepts')
def linear_regression_concepts():
    return render_template('linear_regression_concepts.html')

@app.route('/concepts_logistic')
def concepts_logistic():
    return render_template('conceptsLR.html')


@app.route('/SML')
def SML():
    return render_template('SML.html')

@app.route('/linear_menu')
def linear_menu():
    return render_template('linear_menu.html')

@app.route('/logistic_menu')
def logistic_menu():
    return render_template('logistic_menu.html')

if __name__ == "__main__":
    app.run(debug=True)