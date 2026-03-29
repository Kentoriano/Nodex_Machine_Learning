from flask import Flask, render_template, request
from models.linear_regression import train_model, predict_calories
from models.iris_lda import train_model as train_lda_model, predict_species

app = Flask(__name__)
linear_model = train_model()
lda_model, scaler, accuracy, graph, cm_graph, precision, recall, f1, roc_graph = train_lda_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/use_cases")
def use_cases():
    return render_template("use_cases.html")

@app.route("/use_cases_alien")
def alien():
    return render_template("alien.html")

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

@app.route("/form", methods=["GET", "POST"])
def form():
    result = None
    if request.method == "POST":
        duration = float(request.form["duration"])
        result = predict_calories(linear_model, duration)
    return render_template("form.html", result=result)

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
if __name__ == "__main__":
    app.run(debug=True)