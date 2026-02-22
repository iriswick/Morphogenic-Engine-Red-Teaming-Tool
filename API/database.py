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

def save_evaluation(model_name, phase, evaluation_results):
    record = {
        "model_tested": model_name,
        "phase": phase,
        "DCS": evaluation_results.get("DCS"),
        "HES": evaluation_results.get("HES"),
        "SIS": evaluation_results.get("SIS")
    }
    
    if collection is not None:
        collection.insert_one(record)
        print("Scores successfully saved to MongoDB!")
    else:
        # Just print it to the terminal so you can verify it works
        print(f"MOCK SAVE - {model_name} Phase {phase} Scores: {evaluation_results}")