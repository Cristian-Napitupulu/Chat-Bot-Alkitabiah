from flask import Flask, render_template, request
import os
import bert
import neural_network

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

using_bert = False

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    prompt = request.form["prompt"]
    if using_bert:
        response = bert.get_response(prompt)
    else:
        response = neural_network.get_response(prompt)
    return response


if __name__ == "__main__":
    app.run(debug=True)
