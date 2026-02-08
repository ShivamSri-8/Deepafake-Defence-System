"""
Image preprocessing utilities
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from config import settings


def load_image(image_path: str) -> np.ndarray:
    """Load image from path and return as RGB numpy array"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_for_xception(image: np.ndarray) -> np.ndarray:
    """Preprocess image for Xception model"""
    # Resize to 299x299
    resized = cv2.resize(image, settings.IMAGE_SIZE)
    # Normalize to [-1, 1]
    normalized = (resized.astype(np.float32) / 127.5) - 1.0
    return normalized


def preprocess_for_efficientnet(image: np.ndarray) -> np.ndarray:
    """Preprocess image for EfficientNet model"""
    # Resize to 380x380
    resized = cv2.resize(image, settings.EFFICIENTNET_SIZE)
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def extract_face(image: np.ndarray, padding: float = 0.3) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Extract face from image using MediaPipe (primary) or OpenCV Haar Cascade (fallback)
    Returns cropped face and face info (bounding box, landmarks)
    """
    face_info = None
    face_crop = None
    
    # Method 1: Try MediaPipe
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
            mp_face_detection = mp.solutions.face_detection
            
            with mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=settings.FACE_DETECTION_CONFIDENCE
            ) as face_detector:
                results = face_detector.process(image)
                
                if results.detections:
                    # Get first detected face
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    
                    h, w = image.shape[:2]
                    
                    # Calculate bounding box with padding
                    x = int(max(0, (bbox.xmin - padding * bbox.width) * w))
                    y = int(max(0, (bbox.ymin - padding * bbox.height) * h))
                    width = int(min(w - x, (1 + 2 * padding) * bbox.width * w))
                    height = int(min(h - y, (1 + 2 * padding) * bbox.height * h))
                    
                    # Crop face
                    face_crop = image[y:y+height, x:x+width]
                    
                    face_info = {
                        "bbox": {"x": x, "y": y, "width": width, "height": height},
                        "confidence": detection.score[0],
                        "method": "mediapipe"
                    }
                    return face_crop, face_info
    except Exception as e:
        print(f"MediaPipe detection failed: {e}")

    # Method 2: OpenCV Haar Cascade Fallback
    # Try multiple cascades for better robustness
    cascades = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt2.xml'
    ]
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        for cascade_name in cascades:
            # Load haar cascade
            cascade_path = cv2.data.haarcascades + cascade_name
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if face_cascade.empty():
                continue
                
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Apply padding
                pad_w = int(w * padding)
                pad_h = int(h * padding)
                
                img_h, img_w = image.shape[:2]
                
                x = max(0, x - pad_w)
                y = max(0, y - pad_h)
                w = min(img_w - x, w + 2 * pad_w)
                h = min(img_h - y, h + 2 * pad_h)
                
                face_crop = image[y:y+h, x:x+w]
                
                face_info = {
                    "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "confidence": 0.85,
                    "method": f"opencv_{cascade_name.replace('.xml', '')}"
                }
                print(f"✅ Face detected using OpenCV ({cascade_name})")
                return face_crop, face_info
            
    except Exception as e:
        print(f"OpenCV detection failed: {e}")
        
    return None


def extract_video_frames(video_path: str, max_frames: int = None) -> List[np.ndarray]:
    """
    Extract frames from video at specified sample rate
    """
    max_frames = max_frames or settings.MAX_VIDEO_FRAMES
    sample_rate = settings.VIDEO_FRAME_SAMPLE_RATE
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
            if len(frames) >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frames


def create_batch(images: List[np.ndarray], preprocess_fn) -> np.ndarray:
    """Create a batch of preprocessed images"""
    preprocessed = [preprocess_fn(img) for img in images]
    return np.array(preprocessed)
