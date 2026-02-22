"""
Face index management for fast face matching (FAISS-free version using scipy)
"""
import numpy as np
from scipy.spatial.distance import cosine
from config import SIMILARITY_THRESHOLD, DUPLICATE_THRESHOLD

# Global variables
known_faces_cache = {}
embeddings_cache = []
person_id_mapping = []

def build_faiss_index(known_faces):
    """Build in-memory index from known faces"""
    global known_faces_cache, embeddings_cache, person_id_mapping
    
    known_faces_cache = known_faces.copy()
    embeddings_cache = []
    person_id_mapping = []
    
    for person_id, embeddings in known_faces.items():
        for embedding in embeddings:
            try:
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype='float32')
                else:
                    embedding = np.array(embedding, dtype='float32')
                
                if embedding.ndim == 1 and len(embedding) > 0:
                    embeddings_cache.append(embedding)
                    person_id_mapping.append(person_id)
            except Exception:
                continue
    
    if embeddings_cache:
        print(f"🚀 Index built: {len(embeddings_cache)} embeddings from {len(known_faces)} people")
    else:
        print("⚠️  No valid embeddings found")

def find_matching_person_fast(face_embedding, similarity_threshold=SIMILARITY_THRESHOLD):
    """Find best matching person using scipy distance"""
    global embeddings_cache, person_id_mapping
    
    if not embeddings_cache or len(person_id_mapping) == 0:
        return None, 0
    
    try:
        # Normalize embedding
        query = np.array(face_embedding, dtype='float32')
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Find best match using cosine similarity
        best_similarity = 0
        best_person = None
        best_idx = -1
        
        for idx, stored_embedding in enumerate(embeddings_cache):
            try:
                # Normalize stored embedding
                stored_norm = np.linalg.norm(stored_embedding)
                normalized_stored = stored_embedding / stored_norm if stored_norm > 0 else stored_embedding
                
                # Calculate cosine similarity (dot product of normalized vectors)
                similarity = float(np.dot(query, normalized_stored))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = idx
                    best_person = person_id_mapping[idx]
            except Exception:
                continue
        
        if best_similarity > similarity_threshold:
            return best_person, best_similarity
        
        return None, 0
    
    except Exception as e:
        print(f"❌ Error in find_matching_person_fast: {e}")
        return None, 0

def check_for_duplicate_during_registration(face_embedding, known_faces, threshold=DUPLICATE_THRESHOLD):
    """Check if face is duplicate during registration"""
    global embeddings_cache, person_id_mapping
    
    # Check against cached index
    if embeddings_cache and len(person_id_mapping) > 0:
        try:
            query = np.array(face_embedding, dtype='float32')
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            
            best_similarity = 0
            best_person = None
            
            for idx, stored_embedding in enumerate(embeddings_cache):
                try:
                    stored_norm = np.linalg.norm(stored_embedding)
                    normalized_stored = stored_embedding / stored_norm if stored_norm > 0 else stored_embedding
                    similarity = float(np.dot(query, normalized_stored))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_person = person_id_mapping[idx]
                except Exception:
                    continue
            
            if best_similarity > threshold:
                print(f"🔍 DUPLICATE DETECTED: {best_person} (similarity: {best_similarity:.4f})")
                return True, best_person, best_similarity
        
        except Exception:
            pass
    
    # Manual fallback check
    best_match_id, best_similarity = None, 0
    for person_id, embeddings in known_faces.items():
        for stored_embedding in embeddings:
            try:
                # Normalize for comparison
                query_norm = np.linalg.norm(face_embedding)
                stored_norm = np.linalg.norm(stored_embedding)
                
                if query_norm > 0 and stored_norm > 0:
                    query_normalized = face_embedding / query_norm
                    stored_normalized = np.array(stored_embedding) / stored_norm
                    similarity = float(np.dot(query_normalized, stored_normalized))
                else:
                    similarity = 0
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = person_id
            except Exception:
                continue
    
    if best_similarity > threshold:
        print(f"🔍 DUPLICATE DETECTED: {best_match_id} (similarity: {best_similarity:.4f})")
        return True, best_match_id, best_similarity
    
    return False, None, best_similarity