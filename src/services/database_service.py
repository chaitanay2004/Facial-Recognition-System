from pymongo import MongoClient
from src.models.facial_auth import FacialUser
from typing import Optional
from config import Config
import bcrypt
from datetime import datetime
from bson import ObjectId

class DatabaseService:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.facial_users = self.db[Config.FACIAL_COLLECTION]
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_user(self, user_data: FacialUser) -> bool:
        """Create new user with facial data"""
        try:
            # Check if user already exists
            if self.facial_users.find_one({"username": user_data.username}):
                return False
            
            user_dict = user_data.dict()
            user_dict["password"] = self._hash_password(user_data.password)
            user_dict["created_at"] = datetime.now().isoformat()
            
            result = self.facial_users.insert_one(user_dict)
            return result.acknowledged
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[FacialUser]:
        """Authenticate user with password"""
        try:
            user_data = self.facial_users.find_one({"username": username})
            if user_data and self._verify_password(password, user_data["password"]):
                return FacialUser(**user_data)
            return None
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[FacialUser]:
        """Get user by username"""
        try:
            user_data = self.facial_users.find_one({"username": username})
            if user_data:
                return FacialUser(**user_data)
            return None
        except Exception as e:
            print(f"Error getting user: {e}")
            return None

    # NEW METHOD FOR ADMIN AUTHENTICATION
    def get_user_by_id(self, user_id: str) -> Optional[FacialUser]:
        """Get user by MongoDB ID for admin authentication"""
        try:
            user_data = self.facial_users.find_one({"_id": ObjectId(user_id)})
            if user_data:
                return FacialUser(**user_data)
            return None
        except Exception as e:
            print(f"Error getting user by ID: {e}")
            return None

    # NEW METHOD TO GET ALL USERS FOR ADMIN
    def get_all_users(self) -> list:
        """Get all users for admin dashboard"""
        try:
            users_data = self.facial_users.find({}, {
                "username": 1,
                "email": 1,
                "image_data": 1,
                "facial_embedding": 1,
                "created_at": 1
            })
            
            users = []
            for user_data in users_data:
                # Convert ObjectId to string for JSON serialization
                user_data["_id"] = str(user_data["_id"])
                users.append(user_data)
            
            return users
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []

    # NEW METHOD TO CHECK IF USER HAS FACIAL DATA
    def user_has_facial_data(self, user_id: str) -> bool:
        """Check if user has facial data stored"""
        try:
            user_data = self.facial_users.find_one(
                {"_id": ObjectId(user_id)}, 
                {"image_data": 1}
            )
            return user_data and user_data.get("image_data") is not None
        except Exception as e:
            print(f"Error checking facial data: {e}")
            return False
    
    def update_facial_embedding(self, username: str, embedding: list, image_data: str) -> bool:
        """Update facial embedding for user"""
        try:
            result = self.facial_users.update_one(
                {"username": username},
                {"$set": {
                    "facial_embedding": embedding,
                    "image_data": image_data
                }}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating facial embedding: {e}")
            return False