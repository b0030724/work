import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Import VADER
import matplotlib.pyplot as plt
import pandas as pd
import langid  # Import langid library

# Set up YouTube Data API
# api_key_path = 'your_api_key.json'  # Replace with your API key path
# credentials = service_account.Credentials.from_service_account_file(api_key_path)
api_key = "AIzaSyAe6UJ2_mzs--jjD0879YvfeY6YL4rUZuQ"
youtube = build("youtube", "v3", developerKey=api_key)


# Define the YouTube video ID for which you want to analyze comments
video_id = 'mHiMcv9Md84'  # Replace with your video's ID


# Function to fetch all English comments from the YouTube video
def get_all_english_video_comments(youtube, **kwargs):
    comments = []
   
    while True:
        results = youtube.commentThreads().list(**kwargs).execute()


        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
           
            # Detect the language of the comment using langid
            lang, _ = langid.classify(comment)
            if lang == 'en':
                comments.append(comment)


        # Check if there are more pages of comments
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
        else:
            break


    return comments

# Get all English comments from the YouTube video
comments = get_all_english_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')

# Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(comment) for comment in comments]


# Create a DataFrame for visualisation
data = {'Comments': comments, 'Sentiment Polarity': sentiments}
df = pd.DataFrame(data)


# Print data with no limit
pd.set_option('display.max_rows', None)
print(df)


# Plot sentiment analysis results
plt.figure(figsize=(8, 6))
plt.hist([s['compound'] for s in sentiments], bins=[-1, -0.5, 0, 0.5, 1], color='lightblue')
plt.title(f'Sentiment Analysis for Video {video_id}')
plt.xlabel('Sentiment Polarity (Compound Score)')
plt.ylabel('Number of Comments')
plt.show()
