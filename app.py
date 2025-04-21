from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import re

app = FastAPI(title="AmazonReviewSentiment")

# Load at startup
vectorizer   = joblib.load("tfidf_vectorizer.joblib")
lr_model     = joblib.load("sentiment_model.joblib")
nb_model     = joblib.load("sentiment_nb_model.joblib")

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: int     # 0=negative, 1=positive
    confidence: float   # probability of the predicted class

def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()

# Logistic Regression endpoint
@app.post("/predict/lr", response_model=PredictResponse)
def predict_lr(req: PredictRequest):
    txt   = clean_text(req.text)
    vec   = vectorizer.transform([txt])
    proba = lr_model.predict_proba(vec)[0]
    pred  = int(proba[1] >= 0.5)
    return PredictResponse(prediction=pred, confidence=round(float(proba[pred]), 4))

# Naive Bayes endpoint
@app.post("/predict/nb", response_model=PredictResponse)
def predict_nb(req: PredictRequest):
    txt   = clean_text(req.text)
    vec   = vectorizer.transform([txt])
    proba = nb_model.predict_proba(vec)[0]
    pred  = int(proba[1] >= 0.5)
    return PredictResponse(prediction=pred, confidence=round(float(proba[pred]), 4))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
