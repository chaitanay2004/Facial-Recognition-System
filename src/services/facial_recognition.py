import base64
import cv2
import numpy as np
from typing import Tuple, Optional, List
import os

class FacialRecognitionService:
    def __init__(self):
        # Load OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def encode_face_from_base64(self, base64_image: str) -> Optional[List[float]]:
        """Simple face detection using OpenCV (for registration)"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image)
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                print("Error: Could not decode image")
                return None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with more sensitive parameters
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            print(f"Found {len(faces)} faces in registration image")
            
            if len(faces) > 0:
                # For simplicity, return face coordinates as "embedding"
                x, y, w, h = faces[0]
                print(f"Face detected at: x={x}, y={y}, w={w}, h={h}")
                return [float(x), float(y), float(w), float(h)]
            else:
                print("No face detected in registration image")
                return None
            
        except Exception as e:
            print(f"Error encoding face: {e}")
            return None
    
    def capture_face_from_webcam(self) -> Tuple[Optional[bytes], Optional[List[float]]]:
        """Capture face from webcam using OpenCV"""
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return None, None
            
            print("Looking for face... Press SPACE to capture, Q to quit")
            captured_image = None
            face_coords = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with more sensitive parameters
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Draw face bounding box if face detected
                display_frame = frame.copy()
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face Detected! Press SPACE", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"Face detected at: x={x}, y={y}, w={w}, h={h}")
                else:
                    cv2.putText(display_frame, "No Face Detected", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add instructions
                cv2.putText(display_frame, "Press SPACE to capture face", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press Q to quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Face Capture - Press SPACE when ready', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE to capture
                    if len(faces) > 0:
                        # Encode the frame as JPEG
                        success, buffer = cv2.imencode('.jpg', frame)
                        if success:
                            captured_image = buffer.tobytes()
                            x, y, w, h = faces[0]
                            face_coords = [float(x), float(y), float(w), float(h)]
                            print("Face captured successfully!")
                            print(f"Captured face coordinates: {face_coords}")
                        break
                    else:
                        print("No face detected! Please position your face properly.")
                
                elif key == ord('q'):  # Q to quit
                    print("Face capture cancelled by user")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            return captured_image, face_coords
            
        except Exception as e:
            print(f"Error capturing face from webcam: {e}")
            return None, None
    
    def verify_faces(self, stored_embedding: List[float], live_embedding: List[float]) -> bool:
        """Simple face verification using OpenCV coordinates"""
        try:
            if not stored_embedding or not live_embedding:
                print("Error: Missing embeddings for verification")
                return False
            
            print(f"Stored embedding: {stored_embedding}")
            print(f"Live embedding: {live_embedding}")
            
            # Simple verification based on face size and position
            stored_area = stored_embedding[2] * stored_embedding[3]  # w * h
            live_area = live_embedding[2] * live_embedding[3]
            
            area_ratio = min(stored_area, live_area) / max(stored_area, live_area)
            
            print(f"Face area ratio: {area_ratio}")
            
            # More lenient threshold for demo purposes
            return area_ratio > 0.5  # 50% similarity threshold (more lenient)
            
        except Exception as e:
            print(f"Error verifying faces: {e}")
            return False
    
    def draw_verification_result(self, image_data: bytes, is_authenticated: bool) -> bytes:
        """Draw verification result on captured image"""
        try:
            if image_data is None:
                # Create a result image if no image
                img = np.zeros((200, 400, 3), dtype=np.uint8)
                color = (0, 255, 0) if is_authenticated else (0, 0, 255)
                img[:] = color
                text = "AUTHENTICATED" if is_authenticated else "UNAUTHENTICATED"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = (400 - text_size[0]) // 2
                text_y = (200 + text_size[1]) // 2
                
                cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
                
                success, buffer = cv2.imencode('.jpg', img)
                return buffer.tobytes() if success else b''
            else:
                # Decode the captured image and add result
                image_array = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                color = (0, 255, 0) if is_authenticated else (0, 0, 255)
                text = "AUTHENTICATED" if is_authenticated else "UNAUTHENTICATED"
                
                # Add result overlay
                cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), color, 3)
                
                success, buffer = cv2.imencode('.jpg', img)
                return buffer.tobytes() if success else b''
                
        except Exception as e:
            print(f"Error drawing verification result: {e}")
            return b''

    # NEW METHOD FOR ADMIN AUTHENTICATION
    def capture_live_image_for_admin(self) -> Tuple[Optional[str], Optional[List[float]]]:
        """Capture live image for admin authentication and return base64"""
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return None, None
            
            print("Admin Authentication - Looking for face... Press SPACE to capture, Q to quit")
            captured_base64 = None
            face_coords = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Draw face bounding box if face detected
                display_frame = frame.copy()
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face Detected! Press SPACE", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No Face Detected", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add admin-specific instructions
                cv2.putText(display_frame, "ADMIN AUTHENTICATION", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture face", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press Q to quit", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Admin Authentication - Capture Face', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE to capture
                    if len(faces) > 0:
                        # Encode the frame as base64
                        success, buffer = cv2.imencode('.jpg', frame)
                        if success:
                            captured_base64 = base64.b64encode(buffer).decode('utf-8')
                            x, y, w, h = faces[0]
                            face_coords = [float(x), float(y), float(w), float(h)]
                            print("Admin authentication face captured successfully!")
                        break
                    else:
                        print("No face detected! Please position face properly.")
                
                elif key == ord('q'):  # Q to quit
                    print("Admin authentication cancelled")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            return captured_base64, face_coords
            
        except Exception as e:
            print(f"Error capturing admin authentication face: {e}")
            return None, None

    def get_confidence_score(self, stored_embedding: List[float], live_embedding: List[float]) -> float:
        """Calculate confidence score for face verification"""
        try:
            if not stored_embedding or not live_embedding:
                return 0.0
            
            # Calculate area similarity
            stored_area = stored_embedding[2] * stored_embedding[3]
            live_area = live_embedding[2] * live_embedding[3]
            area_ratio = min(stored_area, live_area) / max(stored_area, live_area)
            
            # Calculate position similarity (normalized)
            height, width = 480, 640  # Assuming standard webcam resolution
            stored_center_x = stored_embedding[0] + stored_embedding[2]/2
            stored_center_y = stored_embedding[1] + stored_embedding[3]/2
            live_center_x = live_embedding[0] + live_embedding[2]/2
            live_center_y = live_embedding[1] + live_embedding[3]/2
            
            position_diff = np.sqrt(
                (stored_center_x - live_center_x)**2 + 
                (stored_center_y - live_center_y)**2
            )
            max_possible_diff = np.sqrt(width**2 + height**2)
            position_similarity = 1.0 - (position_diff / max_possible_diff)
            
            # Combined confidence score
            confidence = (area_ratio + position_similarity) / 2.0
            return round(confidence * 100, 2)  # Return as percentage
            
        except Exception as e:
            print(f"Error calculating confidence score: {e}")
            return 0.0