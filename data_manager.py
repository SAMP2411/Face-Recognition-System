"""
Data management utilities
"""
import json
import os
import time
import numpy as np
from config import KNOWN_FACES_JSON, LOG_FILE

def load_known_faces():
    if not os.path.exists(KNOWN_FACES_JSON):
        return {}
    
    try:
        with open(KNOWN_FACES_JSON, 'r') as f:
            data = json.load(f)
        
        validated_data = {}
        for person_id, embeddings in data.items():
            valid_embeddings = [np.array(emb) for emb in embeddings 
                              if np.array(emb).ndim == 1 and len(emb) > 0]
            if valid_embeddings:
                validated_data[person_id] = valid_embeddings
        
        print(f"📚 Loaded {len(validated_data)} people")
        return validated_data
    except Exception as e:
        print(f"❌ Load error: {e}")
        return {}

def save_known_faces(data):
    try:
        serializable_data = {
            person_id: [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                       for emb in embeddings]
            for person_id, embeddings in data.items()
        }
        
        with open(KNOWN_FACES_JSON, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"💾 Saved {len(serializable_data)} people")
    except Exception as e:
        print(f"❌ Save error: {e}")

def log_event(event_type, person_id, similarity=None):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "person_id": person_id
    }
    if similarity is not None:
        log_entry["similarity"] = similarity
    
    try:
        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception:
        pass

def generate_random_id():
    import random
    return str(random.randint(10000, 99999))

def delete_known_face(person_id):
    """Delete a specific person from the known faces JSON"""
    data = load_known_faces()
    if person_id in data:
        del data[person_id]
        save_known_faces(data)
        print(f"🗑️  Deleted person {person_id}")
        return True
    print(f"⚠️  Person {person_id} not found")
    return False

def clear_known_faces():
    """Remove all known faces (clear the database)"""
    try:
        save_known_faces({})
        print("🗑️  Cleared all known faces")
        return True
    except Exception as e:
        print(f"❌ Failed to clear known faces: {e}")
        return False