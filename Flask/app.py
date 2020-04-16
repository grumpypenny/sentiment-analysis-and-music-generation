"""
This code is for testing flask and the virtual enviroment
"""

from flask import Flask, request, render_template
from model_runner import generate_ABC

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Get input
@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['text']
    generated = generate_ABC(text)
    generated = generated.replace('\n', '<br/>')
    return generated

    # return "DUMMY TEXT LOREM IPSUM"