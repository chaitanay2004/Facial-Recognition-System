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
        print(f"Registration attempt for user: {username}, email: {email}")
        
        # Check if user already exists
        existing_user = db_service.get_user_by_username(username)
        if existing_user:
            print(f"User {username} already exists")
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Read and encode uploaded image
        image_data = await image.read()
        print(f"Image size: {len(image_data)} bytes")
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        print(f"Base64 image length: {len(base64_image)}")
        
        # Extract facial embeddings from uploaded image
        print("Attempting face encoding...")
        facial_embedding = facial_service.encode_face_from_base64(base64_image)
        
        if not facial_embedding:
            print("No face detected in the image")
            raise HTTPException(status_code=400, detail="No face detected in uploaded image. Please ensure your face is clearly visible.")
        
        print(f"Face encoded successfully. Embedding: {facial_embedding}")
        
        # Create user with facial data
        user = FacialUser(
            username=username,
            password=password,
            email=email,
            facial_embedding=facial_embedding,
            image_data=base64_image
        )
        
        success = db_service.create_user(user)
        if not success:
            print("Failed to create user in database")
            raise HTTPException(status_code=500, detail="Failed to create user in database")
        
        print(f"User {username} registered successfully")
        
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
        print(f"Unexpected error during registration: {str(e)}")
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

# NEW ENDPOINT FOR ADMIN AUTHENTICATION
@router.post("/admin/authenticate")
async def admin_authenticate_user(
    user_id: str = Form(...),
    live_image: str = Form(...)  # Base64 encoded live image from webcam
):
    """Admin endpoint to authenticate user using stored image vs live capture"""
    try:
        # Get user from database using user_id
        user = db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if user has stored facial image
        if not user.image_data:
            raise HTTPException(status_code=400, detail="No facial image stored for this user")
        
        print(f"Admin authentication for user: {user.username}")
        print("Comparing stored image with live capture...")
        
        # Extract facial embedding from stored image
        stored_embedding = facial_service.encode_face_from_base64(user.image_data)
        if not stored_embedding:
            raise HTTPException(status_code=400, detail="No face detected in stored image")
        
        # Extract facial embedding from live image
        live_embedding = facial_service.encode_face_from_base64(live_image)
        if not live_embedding:
            raise HTTPException(status_code=400, detail="No face detected in live image")
        
        # Verify faces match
        verification_result = facial_service.verify_faces(stored_embedding, live_embedding)
        
        # Calculate confidence score if available
        confidence = facial_service.get_confidence_score(stored_embedding, live_embedding) if hasattr(facial_service, 'get_confidence_score') else None
        
        return JSONResponse(
            status_code=200,
            content={
                "authenticated": verification_result,
                "confidence": confidence,
                "user": {
                    "id": user_id,
                    "username": user.username,
                    "email": user.email
                },
                "message": "Authentication successful" if verification_result else "Authentication failed - faces don't match"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Admin authentication failed: {str(e)}")

# NEW ENDPOINT TO GET USER FACIAL DATA FOR ADMIN
@router.get("/admin/user/{user_id}/facial-data")
async def get_user_facial_data(user_id: str):
    """Get user's stored facial image data for admin purposes"""
    try:
        user = db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user_id,
            "username": user.username,
            "email": user.email,
            "has_facial_data": user.image_data is not None,
            "image_data": user.image_data if user.image_data else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user facial data: {str(e)}")