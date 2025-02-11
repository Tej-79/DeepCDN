import os
import time
import pandas as pd
import schedule
from googleapiclient.discovery import build

# Set API Key
API_KEY = "AIzaSyAD8PLE7TdBCW_50-1wISDEIbXde1wMkdU"

# Countries and Search Terms
COUNTRIES = ["US", "IN", "GB", "CA", "DE"]  # USA, India, UK, Canada, Germany
SEARCH_TERMS = ["technology", "education", "gaming", "news", "sports"]

# Initialize API
youtube = build("youtube", "v3", developerKey=API_KEY)

def fetch_videos():
    all_data = []
    
    # Fetch Trending Videos
    for country in COUNTRIES:
        request = youtube.videos().list(part="snippet,statistics", chart="mostPopular", regionCode=country, maxResults=50)
        response = request.execute()
        for item in response.get("items", []):
            all_data.append({
                "source": "trending",
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "category": item["snippet"]["categoryId"],
                "view_count": item["statistics"].get("viewCount", 0),
                "like_count": item["statistics"].get("likeCount", 0),
                "comment_count": item["statistics"].get("commentCount", 0),
                "published_at": item["snippet"]["publishedAt"],
                "country": country
            })

    # Fetch Search-based Videos
    for term in SEARCH_TERMS:
        request = youtube.search().list(part="snippet", q=term, type="video", maxResults=50)
        response = request.execute()
        for item in response.get("items", []):
            all_data.append({
                "source": "search",
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "category": item["snippet"]["categoryId"],
                "view_count": 0,  # No direct stats in search API
                "like_count": 0,
                "comment_count": 0,
                "published_at": item["snippet"]["publishedAt"],
                "country": "N/A"
            })

    # Save Data
    df = pd.DataFrame(all_data)
    filename = f"C:/Users/Asus/OneDrive/Desktop/c++/youtube_dataset_{time.strftime('%Y-%m-%d_%H-%M')}.csv"
    # os.makedirs("youtube_data", exist_ok=True)  # Ensure folder exists
    df.to_csv(filename, index=False)
    print(f"✅ Data saved to {filename}")

# Schedule the script every 8 hours
schedule.every(8).hours.do(fetch_videos)

# Run loop
while True:
    schedule.run_pending()
    time.sleep(60)
