import requests
import json

def fetch_and_filter_verses():
    # URL for the raw JSON data from the Bhagavad Gita repository
    url = "https://raw.githubusercontent.com/gita/gita/main/data/verse.json"
    
    print("Fetching verses from GitHub...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch verses. Status code: {response.status_code}")
        return
    
    # Load the JSON data (expected to be a list of dictionaries)
    verses = response.json()
    print(f"Fetched {len(verses)} verses.")
    
    # Filter each verse to include only the required fields for search.
    # Here we keep:
    # - chapter_number: for grouping/fetching by chapter
    # - verse_number: for locating the specific verse
    # - text: the original Sanskrit text (which you might search over)
    # - transliteration: a transliterated version of the verse (optional)
    # - word_meanings: meanings of the words (optional)
    filtered_verses = []
    for verse in verses:
        filtered = {
            "chapter_number": verse.get("chapter_number"),
            "verse_number": verse.get("verse_number"),
            "text": verse.get("text"),
            "transliteration": verse.get("transliteration"),
            "word_meanings": verse.get("word_meanings")
        }
        filtered_verses.append(filtered)
    
    # Write the filtered verses to a new JSON file
    output_file = "filtered_verses.json"
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(filtered_verses, outfile, ensure_ascii=False, indent=2)
    
    print(f"Filtered verses have been written to {output_file}.")

if __name__ == "__main__":
    fetch_and_filter_verses()
