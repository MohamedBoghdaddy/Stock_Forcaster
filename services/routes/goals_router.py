from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, json
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phi2-finetuned")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="./offload"
)
model.eval()

class GoalRequest(BaseModel):
    profile: dict
    advice: dict

@router.post("/api/goals/generate")
async def generate_goal_plan(req: GoalRequest):
    try:
        profile_text = "\n".join(f"{k}: {v}" for k, v in req.profile.items())
        advice_text = "\n".join(req.advice.get("advice", []))

        prompt = f"""
You're an AI financial coach helping users turn advice into actionable goals.

User Profile:
{profile_text}

AI Advice:
{advice_text}

### TASK:
Generate 2-3 long-term goals. For each goal, break it into 2-3 milestones with suggested deadlines.
Respond in JSON format:

[
  {{
    "goal": "Buy a house",
    "milestones": [
      {{ "task": "Save first 20,000 EGP", "target_date": "2025-12" }},
      {{ "task": "Get pre-approved for mortgage", "target_date": "2026-03" }}
    ]
  }},
  ...
]
Only respond with valid JSON.
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.7,
                top_p=0.95
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        json_start = decoded.find("[")
        json_end = decoded.rfind("]") + 1
        parsed = json.loads(decoded[json_start:json_end])
        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
