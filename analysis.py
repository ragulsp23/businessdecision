import gzip
import urllib.request
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Stream Amazon Reviews (limit to, say, 1M reviews for demo)
def stream_amazon_reviews(url, max_reviews=35000000):
    reviews = []
    with urllib.request.urlopen(url) as resp, gzip.GzipFile(fileobj=resp) as f:
        for i, line in enumerate(f):
            if i >= max_reviews:
                break
            try:
                r = json.loads(line.decode("utf-8"))
                if 'reviewText' in r and 'overall' in r:
                    reviews.append({
                        'text': r['reviewText'],
                        'label': 1 if r['overall'] >= 4 else 0
                    })
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(reviews)

# 2. Clean text
def clean_text(s):
    s = s.lower()
    s = re.sub(r'[^a-z\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

# 3. Load, clean, vectorize, train, and export
if __name__ == "__main__":
    print("Downloading data…")
    url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Amazon_Review.json.gz"
    df = stream_amazon_reviews(url)
    df['text'] = df['text'].map(clean_text)
    df.dropna(inplace=True)

    # 4. TF‑IDF + train/test split
    vec = TfidfVectorizer(max_features=5_000)
    X = vec.fit_transform(df['text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train logistic regression
    print("Training LogisticRegression…")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 6. Export artifacts
    joblib.dump(vec, "tfidf_vectorizer.joblib")
    joblib.dump(model, "sentiment_model.joblib")
    print("Saved tfidf_vectorizer.joblib and sentiment_model.joblib")
