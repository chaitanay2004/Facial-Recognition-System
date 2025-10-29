from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import base64
from src.models.facial_auth import FacialUser, FacialLoginRequest
from src.services.facial_recognition import FacialRecognitionService
from src.services.database_service import DatabaseService

router = APIRouter()
facial_service = FacialRecognitionService()
db_service = DatabaseService()

@router.post("/register")
async def register_user(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    image: UploadFile = File(...)
):
    """Register new user with uploaded facial image"""
    try:
        # Check if user already exists
        existing_user = db_service.get_user_by_username(username)
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Read and encode uploaded image
        image_data = await image.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Extract facial embeddings from uploaded image
        facial_embedding = facial_service.encode_face_from_base64(base64_image)
        if not facial_embedding:
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")
        
        # Create user with facial data
        user = FacialUser(
            username=username,
            password=password,
            email=email,
            facial_embedding=facial_embedding,
            image_data=base64_image  # Store the uploaded image
        )
        
        success = db_service.create_user(user)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "User registered successfully", 
                "username": username,
                "face_registered": True
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/login")
async def login_user(login_request: FacialLoginRequest):
    """Login user with credentials + webcam face verification"""
    try:
        # Get user from database
        user = db_service.get_user_by_username(login_request.username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify password first
        if not db_service.authenticate_user(login_request.username, login_request.password):
            raise HTTPException(status_code=401, detail="Invalid password")
        
        # Check if user has facial data
        if not user.facial_embedding:
            raise HTTPException(status_code=400, detail="No facial data registered for this user")
        
        print("Password verified. Starting facial verification...")
        print("Please look at the webcam when prompted")
        
        # Capture face from webcam
        captured_image, live_embedding = facial_service.capture_face_from_webcam()
        
        if not captured_image or not live_embedding:
            raise HTTPException(status_code=400, detail="Failed to capture face from webcam")
        
        # Verify against stored face
        verification_result = facial_service.verify_faces(
            user.facial_embedding, 
            live_embedding
        )
        
        # Create result image
        result_image = facial_service.draw_verification_result(captured_image, verification_result)
        base64_result = base64.b64encode(result_image).decode('utf-8') if result_image else ""
        
        if verification_result:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "Facial authentication successful",
                    "authenticated": True,
                    "result_image": base64_result,
                    "user": {
                        "username": user.username,
                        "email": user.email
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=401,
                content={
                    "message": "Facial authentication failed - Face does not match",
                    "authenticated": False,
                    "result_image": base64_result
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.get("/check-user/{username}")
async def check_user_exists(username: str):
    """Check if user exists in database"""
    user = db_service.get_user_by_username(username)
    if user:
        return {"exists": True, "has_face_data": user.facial_embedding is not None}
    return {"exists": False}