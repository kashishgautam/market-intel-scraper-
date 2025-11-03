"""
analyzer.py
------------
Loads cleaned tweet data and performs:
- TF-IDF vectorization
- Sentiment analysis
- Stock ticker-level signal aggregation
- Memory-efficient sentiment trend plotting
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import os

# Setup logging
logging.basicConfig(
    filename='app.log', level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_data(file_path):
    logging.info(f"Loading cleaned data from {file_path}...")
    return pd.read_parquet(file_path)


def compute_tfidf(data, max_features=500):
    logging.info("Calculating TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(data['cleaned_tweet'])
    return tfidf_matrix, vectorizer


def analyze_sentiment(data):
    logging.info("Performing sentiment analysis on tweets...")
    analyzer = SentimentIntensityAnalyzer()
    data['sentiment_score'] = data['cleaned_tweet'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    return data


def plot_sentiment_over_time(data, sample_size=500, output_path="sentiment_trend.png"):
    logging.info("Plotting sentiment over time...")
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        sample = data.dropna(subset=['timestamp']).sample(min(sample_size, len(data)))
        sample = sample.sort_values(by='timestamp')

        plt.figure(figsize=(10, 5))
        plt.plot(sample['timestamp'], sample['sentiment_score'], marker='o')
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Sentiment Score")
        plt.title("Sentiment Trend Over Time (Sampled)")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Sentiment trend plot saved as {output_path}")
    else:
        logging.warning("Timestamp column not present. Skipping time-based plot.")


def aggregate_signals(data, output_file="ticker_signals.csv"):
    logging.info("Aggregating signals per ticker...")
    exploded = data.explode('tickers')  # one row per ticker
    ticker_signals = exploded.groupby('tickers')['sentiment_score'].agg(
        avg_sentiment='mean',
        tweet_count='size'
    ).reset_index()
    ticker_signals.to_csv(output_file, index=False)
    logging.info(f"Ticker signal summary saved to {output_file}")


def analyzer_pipeline(input_file):
    # Load data
    data = load_data(input_file)

    # Compute sentiment
    data = analyze_sentiment(data)

    # Save sentiment-scored data
    sentiment_csv = "tweets_with_sentiment.csv"
    data.to_csv(sentiment_csv, index=False)
    logging.info(f"Sentiment-scored data saved to {sentiment_csv}")

    # Compute TF-IDF vectors (optional)
    tfidf_matrix, vectorizer = compute_tfidf(data)

    # Plot sentiment trend
    plot_sentiment_over_time(data)

    # Aggregate stock-level signals
    aggregate_signals(data)

    logging.info("Analysis pipeline completed successfully ✅")


if __name__ == "__main__":
    analyzer_pipeline("data/tweets_cleaned.parquet")
