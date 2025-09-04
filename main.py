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
#   - Aly Gutierrez (ID: ########)
#   - Luis Martinez (ID: ########)
#   - Tristan Rodriguez (ID: ########)
#   - Alfredo Peña (ID: 20217083)
#####################################################

from HTMLParser import HTMLParser

def main():
    parser = HTMLParser()

    print("SEARCH ENGINE PRIMED AND READY. (type '/exit' to quit.)")
    while True:
        query = input("Search Term: ").strip()
        if query.startswith("/exit"):
            print("Bye!")
            break

        results = parser.search(query)
        if results:
            print("\n".join(results))
        else:
            print("No Match...")

if __name__ == "__main__":
    main()
