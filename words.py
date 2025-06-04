# Count total words in the text file
with open("wikidata.txt", "r", encoding="utf-8") as file:
    text = file.read()
    total_words = len(text.split())
    print(f"Total words in the file: {total_words}")