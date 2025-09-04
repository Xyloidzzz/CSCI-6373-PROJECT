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
