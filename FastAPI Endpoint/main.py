import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "its", "me",
    "my", "of", "on", "or", "our", "so", "that", "the", "their", "them",
    "there", "they", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "will", "with", "you", "your", "about", "can", "could",
    "would", "should", "dont", "didnt", "im", "ive", "we", "had", "did",
}

SLANG_MAP = {
    "luv": "love", "plz": "please", "pls": "please", "wth": "what the hell",
    "omg": "oh my god", "cant": "cannot", "can't": "cannot", "wanna": "want to",
    "gonna": "going to", "b4": "before", "u": "you", "r": "are", "n't": " not",
}

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#[\w-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [SLANG_MAP.get(w, w) for w in words]
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)


pipeline = joblib.load("LR_pipeline.pkl")
app = FastAPI(title="Airline Sentiment API")

class PredictRequest(BaseModel):
    text: str



@app.post("/predict")
def predict(request: PredictRequest):
    cleaned = preprocess_text(request.text)
    prediction = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]

    return {
        "text": request.text,
        "cleaned_text": cleaned,
        "predicted_sentiment": prediction,
        "probabilities": dict(zip(pipeline.classes_, probabilities.round(4)))
    }



@app.get("/")
def root():
    return {"message": "API is running. Go to /docs to test it."}

