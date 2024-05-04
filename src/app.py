from flask import Flask, render_template, request
import src.bert as bert

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    prompt = request.form["prompt"]
    response = bert.get_response(prompt)
    return response


if __name__ == "__main__":
    app.run(debug=True)
