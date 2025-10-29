import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/accessly")
    DATABASE_NAME = "accessly"
    FACIAL_COLLECTION = "facial_users"
    
    # Facial recognition settings
    FACE_MATCH_TOLERANCE = 0.6
    CAMERA_INDEX = 0
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8001