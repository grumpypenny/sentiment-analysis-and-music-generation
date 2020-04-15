"""
This code is for testing flask and the virtual enviroment
"""

from flask import Flask
from flask import render_template

app = Flask(__name__)

# / is the first directory
@app.route("/")
def hello(name = None):
    return "hello world!!!"