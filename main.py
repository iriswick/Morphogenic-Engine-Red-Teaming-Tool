import json, os, traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from judge import evaluate_llm_response
from database import save_evaluation

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AuditRequest(BaseModel):
    target_api_key: str
    target_model: str

@app.get("/")
async def serve_home(): return FileResponse("Mertt.html")

@app.get("/MerttOutput.html")
async def serve_output(): return FileResponse("MerttOutput.html")

@app.get("/MerttStyles.css")
async def serve_css(): return FileResponse("MerttStyles.css")

@app.post("/run_audit")
async def run_audit_endpoint(req: AuditRequest):
    try:
        # Check for scenarios.json
        if not os.path.exists("scenarios.json"):
            return {"status": "Error", "message": "scenarios.json is missing from the root!"}

        with open("scenarios.json", "r") as file:
            data = json.load(file)

        target_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=req.target_api_key)
        overall_audit_results = []

        for scenario in data["cases"]:
            conversation_history = []
            scenario_results = []
            for index, prompt in enumerate(scenario["prompts"]):
                turn_number = index + 1
                current_phase = ((turn_number - 1) // 3) + 1
                conversation_history.append({"role": "user", "content": prompt})
                
                # Call Target AI
                res = await target_client.chat.completions.create(model=req.target_model, messages=conversation_history)
                ai_answer = res.choices[0].message.content
                conversation_history.append({"role": "assistant", "content": ai_answer})
                
                # Grade and Save (The usual crash points)
                scores = evaluate_llm_response(conversation_history, ai_answer, current_phase)
                save_evaluation(req.target_model, scenario["id"], current_phase, scores, turn_number, prompt, ai_answer, conversation_history)
                
                scenario_results.append({"turn": turn_number, "judges_scores": scores})

            overall_audit_results.append({"scenario_name": scenario["name"], "results": scenario_results})

        return {"status": "Success", "model_tested": req.target_model, "all_results": overall_audit_results}

    except Exception as e:
        # This sends the REAL error to your browser
        return {"status": "Error", "message": str(e), "traceback": traceback.format_exc()}

@app.get("/all_history")
async def get_history():
    from database import get_all_evaluations
    results = get_all_evaluations()
    return {"history": results}

