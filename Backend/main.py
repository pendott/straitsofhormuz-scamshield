import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("scam_model.pkl")

class MessageIn(BaseModel):
    message: str

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/analyze")
def analyze(payload: MessageIn):
    text = payload.message.strip()

    if not text:
        return {
            "verdict": "No message provided",
            "risk_score": 0,
            "risk_level": "Low",
            "scam_type": "SMS/Message Scam",
            "reasons": ["Empty input"],
            "safe_action": "Paste a suspicious message to analyze."
        }

    probs = model.predict_proba([text])[0]
    scam_prob = float(probs[1]) * 100

    if scam_prob >= 70:
        risk_level = "High"
        verdict = "Likely scam"
    elif scam_prob >= 40:
        risk_level = "Medium"
        verdict = "Suspicious"
    else:
        risk_level = "Low"
        verdict = "Likely safe"

    reasons = []
    lowered = text.lower()

    if "http" in lowered or "www" in lowered:
        reasons.append("Contains a link")
    if "urgent" in lowered or "immediately" in lowered:
        reasons.append("Uses urgency language")
    if "otp" in lowered or "password" in lowered or "pin" in lowered:
        reasons.append("Requests sensitive credentials")
    if "bank" in lowered or "police" in lowered or "lhdn" in lowered:
        reasons.append("Uses authority-related language")

    if not reasons:
        reasons.append("Prediction based on wording patterns from trained spam dataset")

    return {
        "verdict": verdict,
        "risk_score": round(scam_prob, 1),
        "risk_level": risk_level,
        "scam_type": "SMS/Message Scam",
        "reasons": reasons[:4],
        "safe_action": "Do not click links or share OTP, PIN, password, or banking details. Verify through the official source."
    }