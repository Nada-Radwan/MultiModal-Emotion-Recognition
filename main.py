# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer

from models.multimodal_emotion import TextEmotionModel
from utils.config import Config

app = FastAPI(title="GoEmotions Text Emotion API")

device = Config.DEVICE

# -------------------------
# Load Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

# -------------------------
# Load Model
# -------------------------
model = TextEmotionModel()
checkpoint = torch.load("checkpoints/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -------------------------
# Emotion Labels
# -------------------------
EMOTION_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

# -------------------------
# Request Schema
# -------------------------
class EmotionRequest(BaseModel):
    text: str


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
def predict(request: EmotionRequest):

    inputs = tokenizer(
        request.text,
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)

        
    # You can choose to return all emotions above a certain threshold or the top-k emotions. 
    # threshold = 0.5
    # predicted_indices = (probs > threshold).nonzero(as_tuple=True)[1].tolist()
    # emotions = [EMOTION_LABELS[i] for i in predicted_indices]

    # Here, we return the top-3 emotions. You can adjust this number as needed.
    top_k = 3
    probs = probs.squeeze()
    top_indices = torch.topk(probs, top_k).indices.tolist()
    emotions = [EMOTION_LABELS[i] for i in top_indices]


    return {
        "text": request.text,
        "predicted_emotions": emotions,
      # "confidence_scores": probs.cpu().tolist()
    }
   