# importing the module
import wikipedia
wikipedia.set_lang("en")

# getting suggestions
result = wikipedia.search("Black holes", results=50)

for i in result:
    x = wikipedia.summary(i)
    try:
        with open("wikidata.txt", "a", encoding="utf-8") as f:  # Specify UTF-8 encoding
            f.write(x + "\n\n")  # Add a newline for better readability
    except Exception as e:
        print(f"An error occurred: {e}")