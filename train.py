import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('/home/parham/nlp-project/Week2/IMDB_Dataset.csv')

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
X = df['review'].apply(clean_text).to_numpy()
y = (df['sentiment'] == 'positive').astype(int).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train_vec, y_train)

print(f"Accuracy: {lr.score(X_test_vec, y_test):.2%}")

with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Model saved!")