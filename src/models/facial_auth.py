
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class FacialUser(BaseModel):
    username: str
    password: str
    email: EmailStr
    facial_embedding: Optional[List[float]] = None  # Numerical face encoding
    image_data: Optional[str] = None  # Base64 stored image
    created_at: Optional[datetime] = None

class FacialLoginRequest(BaseModel):
    username: str
    password: str

class FacialVerificationRequest(BaseModel):
    username: str
    # No image in login request - webcam will capture it
