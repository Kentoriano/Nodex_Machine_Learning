from flask import Flask, render_template, request
import pickle
from models.linear_regression import train_model, predict_calories

app = Flask(__name__)

# Load model
model = train_model()

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

        result = predict_calories(model, duration)

    return render_template("form.html", result=result)
@app.route("/sales")
def sales():
    return render_template("sales.html")
if __name__ == "__main__":
    app.run(debug=True)