#####################################################
# main.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
# Project Part 1
#
# Created: 2025-09-04
# Last Edited: 2025-09-04
#
# Authors:
#   - Aly Gutierrez (ID: 20027057)
#   - Luis Martinez (ID: 20578284)
#   - Tristan Rodriguez (ID: 20499834)
#   - Alfredo Peña (ID: 20217083)
#####################################################

from HTMLParser import HTMLParser

def main():
    parser = HTMLParser()

    print("SEARCH ENGINE PRIMED AND READY. (type '/exit' to quit, '/help' for instructions.)")
    while True:
        query = input("Search Term: ").strip()
        
        if query.startswith("/exit"):
            print("Bye!")
            break
        elif query.startswith("/help"):
            print("Enter one search term to find matching files.")
            print("Enter multiple search terms separated by spaces to find files containing that exact phrase.")
            print("The phrase must appear in the same order in the file.")
            print("Type '/list [n]' to see a list of indexed words (default first 50, or specify n).")
            continue
        elif query.startswith("/list"):
            parts = query.split()
            limit = 50
            if len(parts) > 1 and parts[1].isdigit():
                limit = int(parts[1])

            words = parser.get_indexed_words()
            print(f"Total Indexed Words: {len(words)}")
            for word in words[:limit]:
                print(word)
            if len(words) > limit:
                print(f"... (showing first {limit} only)")
            continue

        results = parser.search(query)
        if results:
            print("\n".join(results))
        else:
            print("No Match...")

if __name__ == "__main__":
    main()
