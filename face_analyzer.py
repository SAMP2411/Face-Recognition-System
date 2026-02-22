"""
InsightFace model initialization and face detection
"""
from insightface.app import FaceAnalysis

class FaceAnalyzer:
    """Main face analysis class using InsightFace"""
    
    def __init__(self):
        """Initialize InsightFace model"""
        print("🔧 Initializing Face Recognition Model...")
        self.app = FaceAnalysis(name='buffalo_sc')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ Face Recognition Model Ready!")
    
    def get_faces(self, frame):
        """Detect and analyze faces in frame"""
        try:
            faces = self.app.get(frame)
            return faces
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            return []
    
    def get_face_embedding(self, frame):
        """Get embedding for the first detected face"""
        faces = self.get_faces(frame)
        if faces:
            return faces[0].embedding
        return None
    
    def get_best_face(self, frame, min_size=80):
        """Get the best quality face from detected faces"""
        faces = self.get_faces(frame)
        if not faces:
            return None
        
        # Filter by minimum size and select largest
        valid_faces = []
        for face in faces:
            bbox = face.bbox.astype(int)
            face_width = bbox[2] - bbox[0]
            if face_width >= min_size:
                valid_faces.append((face, face_width))
        
        if valid_faces:
            # Return face with largest width (closest to camera)
            return max(valid_faces, key=lambda x: x[1])[0]
        
        return None