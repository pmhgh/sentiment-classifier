# Movie Review Sentiment Classifier

A machine learning model that predicts if a movie review is positive or negative using Logistic Regression and TF-IDF.

## Results
- **Accuracy:** 89.75%
- **Precision:** 89.00%
- **Recall:** 90.89%
- **F1-Score:** 89.94%

## Files
- `train.py` - Trains model on IMDB dataset (50K reviews)
- `app.py` - Streamlit web UI for predictions
- `sentiment_model.pkl` - Trained Logistic Regression model
- `vectorizer.pkl` - TF-IDF vectorizer

## How to Run

1. Install dependencies:
```bash
   pip install -r requirements.txt
```

2. Run the app:
```bash
   streamlit run app.py
```

3. Open `http://localhost:8501` in browser

4. Paste a movie review, click predict

## Dataset
IMDB Dataset of 50K Movie Reviews (balanced positive/negative)

## Technologies
- scikit-learn (Logistic Regression, TF-IDF)
- Streamlit (Web UI)
- pandas, numpy

