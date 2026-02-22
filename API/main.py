import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # NEW: Required for serving files
from pydantic import BaseModel
from openai import AsyncOpenAI

# Import your custom functions
from judge import evaluate_llm_response
from database import save_evaluation

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuditRequest(BaseModel):
    target_api_key: str
    target_model: str

# --- FRONTEND ROUTES ---
@app.get("/")
async def serve_home():
    return FileResponse("Mertt.html")

@app.get("/Mertt.html")
async def serve_mertt_html():
    return FileResponse("Mertt.html")

@app.get("/MerttOutput.html")
async def serve_output():
    return FileResponse("MerttOutput.html")

@app.get("/MerttStyles.css")
async def serve_css():
    return FileResponse("MerttStyles.css")

# --- BACKEND ROUTES ---
@app.get("/scenarios")
async def get_scenarios():
    with open("scenarios.json", "r") as file:
        data = json.load(file)
    return data

@app.post("/run_audit")
async def run_audit_endpoint(req: AuditRequest):
    with open("scenarios.json", "r") as file:
        data = json.load(file)

    target_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=req.target_api_key
    )
    
    overall_audit_results = []

    for scenario in data["cases"]:
        conversation_history = []
        scenario_results = []

        for index, prompt in enumerate(scenario["prompts"]):
            turn_number = index + 1
            current_phase = ((turn_number - 1) // 3) + 1
            conversation_history.append({"role": "user", "content": prompt})
            
            try:
                target_response = await target_client.chat.completions.create(
                    model=req.target_model,
                    messages=conversation_history,
                    temperature=0.7 
                )
                ai_answer = target_response.choices[0].message.content
            except Exception as e:
                scenario_results.append({"error": str(e), "failed_at_turn": turn_number})
                break 
            
            conversation_history.append({"role": "assistant", "content": ai_answer})
            scores = evaluate_llm_response(conversation_history, ai_answer, current_phase)
            save_evaluation(req.target_model, scenario["id"], current_phase, scores, turn_number, prompt, ai_answer, conversation_history)
            
            scenario_results.append({
                "turn": turn_number,
                "phase": current_phase,
                "prompt": prompt,
                "ai_response": ai_answer,
                "judges_scores": scores
            })

        overall_audit_results.append({
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "theme": scenario["theme"],
            "results": scenario_results
        })

    return {
        "status": "Full Audit Complete",
        "model_tested": req.target_model,
        "total_scenarios_run": len(overall_audit_results),
        "all_results": overall_audit_results
    }
