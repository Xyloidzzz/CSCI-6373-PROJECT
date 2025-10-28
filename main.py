#####################################################
# main.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
#
# Created: 2025-09-04
# Last Edited: 2025-10-09
#
# Authors:
#   - Ariana Gutierrez
#   - Luis Martinez
#   - Tristan Rodriguez
#   - Alfredo Peña
#####################################################

from flask import Flask, render_template, request
from HTMLParser import HTMLParser

parser = HTMLParser()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    q = (request.args.get("q") or "").strip()
    results = None
    if q:
        results = parser.search(q) or []
    return render_template("index.html", q=q, results=results, links=parser.links)

if __name__ == "__main__":
    app.run(debug=True)
