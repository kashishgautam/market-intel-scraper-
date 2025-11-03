import snscrape.modules.twitter as sntwitter
import pandas as pd

def scrape_tweets(hashtags, limit_per_tag=500):
    tweets_list = []
    for tag in hashtags:
        query = f"{tag} since:2024-01-01 until:2024-01-02"  # last 24 hours
        scraped_tweets = sntwitter.TwitterSearchScraper(query).get_items()
        
        for i, tweet in enumerate(scraped_tweets):
            if i > limit_per_tag:
                break
            tweets_list.append([
                tweet.user.username,
                tweet.date,
                tweet.rawContent,
                tweet.likeCount,
                tweet.retweetCount
            ])
    return pd.DataFrame(tweets_list, columns=["username", "timestamp", "content", "likes", "retweets"])


hashtags = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
tweets_df = scrape_tweets(hashtags, 600)
tweets_df.to_csv("data/tweets.csv", index=False)
print("✅ Saved tweets to data/tweets.csv")
