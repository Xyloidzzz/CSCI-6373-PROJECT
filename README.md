# CSCI-6373-PROJECT

CSCI 6373 - Information Retrieval and Web Search Group Project

## Project Members

- Ariana Gutierrez
- Luis Martinez
- Tristan Rodriguez
- Alfredo Peña

## Project Description

Search Engine Website precursor that indexes HTML files from a ZIP archive and allows users to search for terms within those files.

## How to Run

1. Ensure you have Python installed on your machine.
2. Install required packages: `pip install -r requirements.txt`
3. Extract all files from rhf.zip into static folder
4. Run the script: `python main.py`
5. Enter search terms when prompted to find relevant HTML file names.

Your search engine will be accessible at: `http://localhost:5000/`

## Supported Search Types

- Boolean Search: Use `and`, `or`, and parentheses for complex queries.
- Phrase Search: Enclose phrases in double quotes (e.g., `"information retrieval"`).
- Ranked Vector Search: Enter terms normally for ranked results.
