from flask import Flask, render_template, request
from translate import Translator
import bert

app = Flask(__name__)

target_language = "id"  # "id" is the ISO 639-1 language code for Indonesian
translator_object = Translator(to_lang=target_language)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    prompt = request.form["prompt"]
    # translation = translator_object.translate(prompt)
    # response = translation
    response = bert.get_response(prompt)
    return response


if __name__ == "__main__":
    app.run(debug=True)
