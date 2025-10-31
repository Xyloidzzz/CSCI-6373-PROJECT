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

from flask import Flask, render_template, request, send_file
from HTMLParser import HTMLParser
import zipfile
from io import BytesIO

parser = HTMLParser()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    q = (request.args.get("q") or "").strip()
    results = None
    if q:
        results = parser.search(q) or []
    return render_template("index.html", q=q, results=results, links=parser.links, titles=parser.titles, snippets=parser.snippets)

@app.route("/doc/<path:filepath>")
def serve_doc(filepath):
    try:
        with zipfile.ZipFile("rhf.zip", 'r') as z:
            file_path = f"rhf/{filepath}"
            if file_path in z.namelist():
                file_data = z.read(file_path)
                return send_file(
                    BytesIO(file_data),
                    mimetype='text/html',
                    as_attachment=False
                )
            else:
                return "File not found", 404
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
