# processing/ai_studio_client.py
import os
import json
import requests

AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
API_KEY = os.getenv("AI_STUDIO_API_KEY")

SYSTEM_PROMPT = (
    "You are an AI hydrogeologist and data interpreter for the Borehole Exploration Analytics (BH-EA) system. "
    "You receive metadata JSON files and contour images from ERT/pumping analyses. "
    "Tasks: (1) interpret hydro meaning (yield trends, transmissivity zones, anomalies), "
    "(2) summarize key metrics (Avg Yield/Cost/Transmissivity/Storage/Efficiency), "
    "(3) identify Goldilocks (🟩) and Trouble (🟥) zones, "
    "(4) give actionable optimization recommendations. Keep it concise and structured."
)

def analyze_with_ai_studio(metadata_url: str, report_url: str, dataset_summary: str):
    if not API_KEY:
        return {"error": "AI_STUDIO_API_KEY is not set"}

    user_prompt = f"""
Input metadata file:
{metadata_url}

Input contour report:
{report_url}

Dataset summary (JSON):
{dataset_summary}

Output strictly as compact JSON with these keys:
{{
  "interpretation_summary": "string",
  "goldilocks_sites": ["optional list of site labels or indices"],
  "trouble_sites": ["optional list of site labels or indices"],
  "recommendations": ["bullet items"]
}}
"""

    payload = {
        "contents": [
            {"parts": [{"text": f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}"}]},
            {"parts": [{"text": f"USER PROMPT:\n{user_prompt}"}]}
        ]
    }
    params = {"key": API_KEY}
    headers = {"Content-Type": "application/json"}

    r = requests.post(AI_STUDIO_URL, params=params, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        return {"error": f"AI Studio HTTP {r.status_code}", "detail": r.text}

    try:
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        # The model returns JSON text; parse it—if it fails, pass raw text back.
        try:
            return json.loads(text)
        except Exception:
            return {"raw_text": text}
    except Exception as e:
        return {"error": "Unexpected AI response format", "detail": str(e)}
