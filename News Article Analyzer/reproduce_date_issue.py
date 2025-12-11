from gnews import GNews
import json

def test_gnews_date():
    google = GNews(max_results=1)
    results = google.get_news("technology")
    
    if results:
        print("Raw Article Keys:", results[0].keys())
        print("Published Date Value:", results[0].get('published date'))
        print("Publisher:", results[0].get('publisher'))
    else:
        print("No results found")

if __name__ == "__main__":
    test_gnews_date()
