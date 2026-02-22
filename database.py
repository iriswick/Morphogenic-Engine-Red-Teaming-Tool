import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") 

if not MONGO_URI or "your-cluster" in MONGO_URI:
    print("WARNING: No valid MONGO_URI found. Running in MOCK database mode.")
    collection = None
else:
    db_client = MongoClient(MONGO_URI)
    db = db_client["mertt_database"]
    collection = db["evaluations"]


def save_evaluation(model_name, scenario_id, phase, evaluation_results, turn, prompt, ai_response, history_array):
    record = {
        "model_tested": model_name,
        "scenario_id": scenario_id,
        "phase": phase,
        "turn": turn,
        "user_prompt": prompt,
        "ai_response": ai_response,
        "conversation_history": history_array,
        
        "gpt_4o_mini_scores": evaluation_results.get("gpt_4o_mini", {})
    }
    
    if collection is not None:
        collection.insert_one(record)
        print(f"Scores & Logs for {scenario_id} (Turn {turn}) successfully saved to MongoDB!")
    else:
        print(f"MOCK SAVE - {model_name} | {scenario_id} | Turn {turn} | Phase {phase}")

def get_all_evaluations():
    """Retrieves all stored audit records from the database."""
    if collection is not None:
        # We convert the cursor to a list and exclude the MongoDB '_id' field
        return list(collection.find({}, {"_id": 0}))
    return []
