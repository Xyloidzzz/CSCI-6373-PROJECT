#####################################################
# main.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
#
# Created: 2025-09-04
# Last Edited: 2025-11-21
#
# Authors:
#   - Ariana Gutierrez
#   - Luis Martinez
#   - Tristan Rodriguez
#   - Alfredo Peña
#####################################################

from flask import Flask, render_template, request, send_file
import os
from dotenv import load_dotenv
from HTMLParser import HTMLParser
import zipfile
from io import BytesIO

load_dotenv()
parser = HTMLParser()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    q = (request.args.get("q") or "").strip()
    mode = request.args.get("mode") or "reformulation"  # default to reformulation
    results = None
    reformulation_info = None
    recommendation_info = None

    if q:
        search_result = parser.search(q, mode=mode)

        # check if result is from content-based recommendation
        if isinstance(search_result, dict) and 'recommended_results' in search_result:
            results = search_result['merged_results']
            recommendation_info = {
                'original': set(search_result['original_results']),
                'recommended': set(search_result['recommended_results']),
                'original_list': search_result['original_results'],
                'recommended_list': search_result['recommended_results'],
                'has_recommendations': len(search_result['recommended_results']) > 0,
                'query_terms': search_result.get('query_terms', []),
                'top_docs_used': search_result.get('top_docs_used', []),
                'recommendation_reasons': search_result.get('recommendation_reasons', {})
            }
        # check if result is from query reformulation
        elif isinstance(search_result, dict):
            results = search_result['merged_results']
            reformulation_info = {
                'original': set(search_result['original_results']),
                'expanded': [doc for doc in search_result['merged_results'] if doc not in search_result['original_results']],
                'original_list': search_result['original_results'],
                'reformulated_query': search_result['reformulated_query'],
                'expansion_terms': search_result['expansion_terms'],
                'has_expansion': len(search_result['expansion_terms']) > 0
            }
        else:
            # boolean or phrase search returns simple list
            results = search_result or []

    return render_template(
        "index.html",
        q=q,
        mode=mode,
        results=results,
        links=parser.links,
        titles=parser.titles,
        snippets=parser.snippets,
        reformulation=reformulation_info,
        recommendation=recommendation_info
    )

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
    app.run(debug=os.environ.get("DEBUG"))
