#####################################################
# main.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
#
# Created: 2025-09-04
# Last Edited: 2025-11-15
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
    results = None
    reformulation_info = None
    comparison_rows = None   # <-- will hold (old, new) pairs for the table

    if q:
        search_result = parser.search(q)

        # check if result is from query reformulation
        if isinstance(search_result, dict):
            # lists of filenames
            original_list = search_result.get("original_results", []) or []
            merged_list = search_result.get("merged_results", []) or []

            # what we use as the main "results" list on the page
            results = merged_list

            # info block for reformulation
            reformulation_info = {
                'original': original_list,  # keep as list (not set) so we preserve order
                'reformulated_query': search_result.get('reformulated_query', ""),
                'expansion_terms': search_result.get('expansion_terms', []),
                'has_expansion': len(search_result.get('expansion_terms', [])) > 0
            }

            # ---- build side-by-side table: original vs merged ----
            max_len = max(len(original_list), len(merged_list))
            rows = []
            for i in range(max_len):
                old_val = original_list[i] if i < len(original_list) else ""
                new_val = merged_list[i] if i < len(merged_list) else ""
                rows.append((old_val, new_val))
            comparison_rows = rows

        else:
            # boolean or phrase search returns simple list
            results = search_result or []
            # no reformulation info, so no side-by-side comparison
            comparison_rows = None

    return render_template(
        "index.html",
        q=q,
        results=results,
        links=getattr(parser, "links", {}),      # in case these attrs exist
        titles=getattr(parser, "titles", {}),
        snippets=getattr(parser, "snippets", {}),
        reformulation=reformulation_info,
        comparison_rows=comparison_rows          # <-- NEW
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

