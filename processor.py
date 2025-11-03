"""
processor.py
-------------
Processes tweets related to Indian stocks from the input CSV file,
cleans and extracts useful features, and stores the result as a Parquet file.
"""

import pandas as pd
import re
import logging

# Setup logging
logging.basicConfig(
    filename='app.log', level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def clean_text(text):
    """Remove extra spaces, newlines, and non-ASCII characters."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace
    text = text.strip()
    return text.encode('utf-8', 'ignore').decode('utf-8')


def extract_hashtags(text):
    """Extract hashtags like #Nifty, #Sensex, etc."""
    return re.findall(r"#(\w+)", text)


def extract_tickers(text):
    """Extract tickers formatted as $ICICIBANK.NSE or $TCS.NSE."""
    return re.findall(r"\$(\w+\.\w+)", text)


def process_tweets(input_file, output_file):
    logging.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    logging.info("Cleaning text in 'tweet' column...")
    df['cleaned_tweet'] = df['tweet'].apply(clean_text)
    
    logging.info("Extracting hashtags and stock tickers...")
    df['hashtags'] = df['cleaned_tweet'].apply(extract_hashtags)
    df['tickers'] = df['cleaned_tweet'].apply(extract_tickers)
    
    logging.info("Adding word and hashtag counts...")
    df['word_count'] = df['cleaned_tweet'].apply(lambda x: len(x.split()))
    df['hashtag_count'] = df['hashtags'].apply(len)
    
    logging.info("Dropping duplicate tweets...")
    df.drop_duplicates(subset=['cleaned_tweet'], inplace=True)
    
    logging.info(f"Saving processed data to {output_file}...")
    df.to_parquet(output_file, index=False)
    
    logging.info("Processing complete! ✅")


if __name__ == "__main__":
    process_tweets("data/tweets.csv", "data/tweets_cleaned.parquet")
