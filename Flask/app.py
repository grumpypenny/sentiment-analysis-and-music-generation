"""
This code is for testing flask and the virtual enviroment
"""

from flask import Flask, request, render_template, redirect, url_for
from model_runner import generate_ABC

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Get input
@app.route('/submit', methods=['POST'])
def submit():
 
    text = request.form['text']
    if text:
        generated = generate_ABC(text)
        generated = generated.replace('\n', '<br/>')
        return generated

    return render_template('index.html')

@app.route('/redirection', methods=['GET', 'POST'])
def redirection():
    if 'about' in request.form:
        return redirect(url_for('about'))

    elif 'usage' in request.form:
        return redirect(url_for('usage'))

    return render_template('index.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/usage', methods=['GET', 'POST'])
def usage():
    return render_template('usage.html')

