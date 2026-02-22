import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# Import your custom functions
from judge import evaluate_llm_response
from database import save_evaluation

app = FastAPI()

# --- CORS SETUP: Allows your HTML frontend to talk to this API ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this. For a hackathon, let it ride.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuditRequest(BaseModel):
    target_api_key: str
    target_model: str
    scenario_id: str 

@app.get("/scenarios")
async def get_scenarios():
    """Reads the scenarios.json file and sends it to the frontend."""
    with open("scenarios.json", "r") as file:
        data = json.load(file)
    return data

@app.post("/run_audit")
async def run_audit_endpoint(req: AuditRequest):
    # 1. Load the requested scenario
    with open("scenarios.json", "r") as file:
        data = json.load(file)
    
    scenario = next((case for case in data["cases"] if case["id"] == req.scenario_id), None)
    if not scenario:
        return {"error": "Scenario not found"}

    # --- THE MAGIC ROUTER ---
    # 2. Check if the user wants to test a Gemini model or an OpenRouter model
    if "gemini" in req.target_model.lower():
        # Use Google's Native OpenAI Compatibility Endpoint
        target_client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=req.target_api_key
        )
    else:
        # Use OpenRouter
        target_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=req.target_api_key
        )
    # ------------------------
    
    conversation_history = []
    audit_results = []

    # 3. The 12-Turn Conversation Loop
    for index, prompt in enumerate(scenario["prompts"]):
        turn_number = index + 1
        current_phase = ((turn_number - 1) // 3) + 1
        
        conversation_history.append({"role": "user", "content": prompt})
        
        # --- Added Try/Catch for Hackathon Resilience ---
        try:
            # Talk to the Target LLM (Works for both OpenRouter AND Native Gemini!)
            target_response = await target_client.chat.completions.create(
                model=req.target_model,
                messages=conversation_history,
                temperature=0.7
            )
            ai_answer = target_response.choices[0].message.content
        except Exception as e:
            print(f"API Error at turn {turn_number}: {e}")
            # Gracefully return whatever results we successfully gathered before the crash
            return {
                "status": "Partial Audit Complete (API Error)",
                "error_details": str(e),
                "failed_at_turn": turn_number,
                "model_tested": req.target_model,
                "scenario_tested": scenario["name"],
                "results": audit_results 
            }
        # ------------------------------------------------
        
        conversation_history.append({"role": "assistant", "content": ai_answer})
        history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        
        # 4. Grade using BOTH Judges
        scores = evaluate_llm_response(history_string, ai_answer, current_phase)
        
        # 5. Save to database
        save_evaluation(req.target_model, current_phase, scores)
        
        audit_results.append({
            "turn": turn_number,
            "phase": current_phase,
            "prompt": prompt,
            "ai_response": ai_answer,
            "judges_scores": scores
        })

    return {
        "status": "Audit Complete",
        "model_tested": req.target_model,
        "scenario_tested": scenario["name"],
        "results": audit_results
    }