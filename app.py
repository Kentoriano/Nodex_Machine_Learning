from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
