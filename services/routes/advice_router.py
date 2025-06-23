from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from datetime import datetime
from db import get_db
import jwt, os
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")

class AdvicePayload(BaseModel):
    user_id: str
    profile_snapshot: dict
    tips: dict

@router.post("/api/advice/save")
async def save_advice(payload: AdvicePayload):
    db = get_db()
    payload_dict = payload.dict()
    payload_dict["createdAt"] = datetime.utcnow()
    db.advice.insert_one(payload_dict)
    return {"message": "Advice saved"}

@router.get("/api/advice/history")
async def get_advice_history(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = auth.split(" ")[1]
    decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    user_id = decoded.get("_id")

    db = get_db()
    history = list(db.advice.find({"user_id": user_id}).sort("createdAt", -1))
    for h in history:
        h["_id"] = str(h["_id"])
    return history
