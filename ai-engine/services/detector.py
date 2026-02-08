"""
Deepfake Detection Service
Implements multi-model ensemble for deepfake detection
"""
import numpy as np
import os
from typing import Optional, List, Dict, Any
import random

from config import settings
from models.schemas import (
    DetectionResult,
    ModelPrediction,
    ConfidenceInterval
)
from utils.preprocessing import (
    load_image,
    preprocess_for_xception,
    preprocess_for_efficientnet,
    extract_face,
    extract_video_frames
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DeepfakeDetector:
    """
    Multi-model ensemble deepfake detector.
    Combines Xception, EfficientNet-B4, and CNN+LSTM predictions.
    """
    
    def __init__(self):
        self.models_loaded = False
        self.xception_model = None
        self.efficientnet_model = None
        self.lstm_model = None
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained model weights - loads each model independently"""
        import tensorflow as tf
        
        loaded_count = 0
        
        # Load Xception model
        if os.path.exists(settings.XCEPTION_MODEL_PATH):
            try:
                logger.info("Loading Xception model...")
                self.xception_model = tf.keras.models.load_model(settings.XCEPTION_MODEL_PATH)
                loaded_count += 1
                logger.info("✅ Xception model loaded")
            except Exception as e:
                logger.error(f"❌ Error loading Xception model: {str(e)}")
        else:
            logger.warning(f"⚠️ Xception model not found at {settings.XCEPTION_MODEL_PATH}")
        
        # Load EfficientNet model
        if os.path.exists(settings.EFFICIENTNET_MODEL_PATH):
            try:
                logger.info("Loading EfficientNet model...")
                self.efficientnet_model = tf.keras.models.load_model(settings.EFFICIENTNET_MODEL_PATH)
                loaded_count += 1
                logger.info("✅ EfficientNet model loaded")
            except Exception as e:
                logger.error(f"❌ Error loading EfficientNet model: {str(e)}")
        else:
            logger.warning(f"⚠️ EfficientNet model not found at {settings.EFFICIENTNET_MODEL_PATH}")
        
        # Load LSTM model
        if os.path.exists(settings.LSTM_MODEL_PATH):
            try:
                logger.info("Loading CNN+LSTM model...")
                self.lstm_model = tf.keras.models.load_model(settings.LSTM_MODEL_PATH)
                loaded_count += 1
                logger.info("✅ CNN+LSTM model loaded")
            except Exception as e:
                logger.error(f"❌ Error loading CNN+LSTM model: {str(e)}")
        else:
            logger.warning(f"⚠️ CNN+LSTM model not found at {settings.LSTM_MODEL_PATH}")
        
        # Set loaded status - at least one model is enough for real predictions
        if loaded_count > 0:
            self.models_loaded = True
            logger.info(f"✅ {loaded_count}/3 models loaded successfully")
        else:
            logger.warning("⚠️ No model weights found. Running in simulation mode.")
            self.models_loaded = False
    
    async def detect_image(self, image_path: str) -> DetectionResult:
        """
        Detect deepfake in a single image
        """
        logger.info(f"Analyzing image: {image_path}")
        
        try:
            # Load and preprocess image
            image = load_image(image_path)
            
            # Extract face
            face_result = extract_face(image)
            face_detected = face_result is not None
            
            if face_result:
                face_image, face_info = face_result
                method = face_info.get('method', 'unknown')
                print(f"✅ FACE DETECTED! Method: {method}, Confidence: {face_info.get('confidence', 0)}")
                logger.info(f"Face detected with method: {method}, confidence: {face_info.get('confidence', 0):.2f}")
            else:
                # Use full image if no face detected
                face_image = image
                print("❌ NO FACE DETECTED - Using full image")
                logger.warning("No face detected, using full image")
            
            # Get predictions from each model
            if self.models_loaded:
                predictions = await self._get_real_predictions(face_image)
            else:
                predictions = self._get_simulated_predictions()
            
            # Ensemble voting with weights
            fake_probability = self._ensemble_vote(predictions)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(predictions)
            
            # Determine risk level
            risk_level = self._determine_risk_level(fake_probability)
            
            # Build result
            result = DetectionResult(
                is_fake=fake_probability >= settings.FAKE_THRESHOLD,
                fake_probability=round(fake_probability, 4),
                confidence=round(self._calculate_confidence(predictions), 4),
                confidence_interval=confidence_interval,
                risk_level=risk_level,
                model_predictions=predictions,
                face_detected=face_detected,
                notes=self._generate_notes(fake_probability, face_detected, predictions)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            raise
    
    async def detect_video(self, video_path: str) -> DetectionResult:
        """
        Detect deepfake in a video by analyzing sampled frames
        and temporal consistency using CNN+LSTM
        """
        logger.info(f"Analyzing video: {video_path}")
        
        try:
            # Extract frames from video
            frames = extract_video_frames(video_path)
            logger.info(f"Extracted {len(frames)} frames from video")
            
            if len(frames) == 0:
                raise ValueError("No frames could be extracted from video")
            
            model_predictions = []
            notes = [f"Analyzed {len(frames)} frames from video"]
            
            # === Per-Frame Analysis (Xception + EfficientNet) ===
            frame_results = []
            for i, frame in enumerate(frames):
                face_result = extract_face(frame)
                if face_result:
                    face_image, _ = face_result
                else:
                    face_image = frame
                
                if self.models_loaded:
                    predictions = await self._get_real_predictions(face_image)
                else:
                    predictions = self._get_simulated_predictions()
                
                frame_fake_prob = self._ensemble_vote(predictions)
                frame_results.append(frame_fake_prob)
            
            # Aggregate frame results
            frame_avg_prob = np.mean(frame_results)
            frame_std_dev = np.std(frame_results)
            
            model_predictions.append(ModelPrediction(
                model_name="Frame Analysis (Xception + EfficientNet)",
                fake_probability=round(frame_avg_prob, 4),
                confidence=round(max(0.1, 1 - frame_std_dev), 4),
                weight=0.6  # 60% weight for frame analysis
            ))
            
            notes.append(f"Frame-to-frame variance: {frame_std_dev:.4f}")
            
            # === Temporal Analysis (CNN+LSTM) ===
            lstm_probability = None
            
            if self.models_loaded and self.lstm_model is not None:
                try:
                    # Prepare frame sequence for LSTM
                    lstm_probability = await self._get_lstm_prediction(frames)
                    
                    model_predictions.append(ModelPrediction(
                        model_name="Temporal Analysis (CNN+LSTM)",
                        fake_probability=round(lstm_probability, 4),
                        confidence=round(0.85, 4),  # LSTM typically has high confidence
                        weight=0.4  # 40% weight for temporal analysis
                    ))
                    
                    notes.append("Temporal consistency analysis performed with LSTM")
                    
                except Exception as e:
                    logger.warning(f"LSTM analysis failed, using frame analysis only: {str(e)}")
                    notes.append("⚠️ Temporal analysis unavailable, using frame analysis only")
            else:
                # Simulated temporal analysis
                if not self.models_loaded:
                    # In simulation mode, generate fake LSTM prediction
                    import random
                    lstm_probability = frame_avg_prob + random.uniform(-0.1, 0.1)
                    lstm_probability = max(0, min(1, lstm_probability))
                    
                    model_predictions.append(ModelPrediction(
                        model_name="Temporal Analysis (CNN+LSTM)",
                        fake_probability=round(lstm_probability, 4),
                        confidence=round(random.uniform(0.75, 0.90), 4),
                        weight=0.4
                    ))
                    notes.append("⚠️ Running in simulation mode - results are for demonstration only")
            
            # === Ensemble Final Probability ===
            fake_probability = self._ensemble_vote(model_predictions)
            
            # High variance between frames is suspicious
            if frame_std_dev > 0.2:
                notes.append("⚠️ High variance detected between frames - possible manipulation boundary")
            
            # If LSTM and frame analysis disagree significantly, note it
            if lstm_probability is not None:
                disagreement = abs(lstm_probability - frame_avg_prob)
                if disagreement > 0.25:
                    notes.append(f"⚠️ Frame and temporal analysis disagree by {disagreement:.0%}")
            
            # Calculate confidence interval
            all_probs = frame_results + ([lstm_probability] if lstm_probability else [])
            overall_std = np.std(all_probs)
            
            confidence_interval = ConfidenceInterval(
                lower=max(0, fake_probability - 1.96 * overall_std),
                upper=min(1, fake_probability + 1.96 * overall_std),
                confidence_level=0.95
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(fake_probability)
            
            # Build result
            result = DetectionResult(
                is_fake=fake_probability >= settings.FAKE_THRESHOLD,
                fake_probability=round(fake_probability, 4),
                confidence=round(self._calculate_confidence(model_predictions), 4),
                confidence_interval=confidence_interval,
                risk_level=risk_level,
                model_predictions=model_predictions,
                face_detected=True,
                notes=notes
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Video detection error: {str(e)}")
            raise
    
    async def _get_lstm_prediction(self, frames: List[np.ndarray]) -> float:
        """Get temporal prediction from LSTM model"""
        import cv2
        
        # Prepare frame sequence (resize to 224x224 for EfficientNet base)
        num_frames = 20  # Match training sequence length
        
        if len(frames) < num_frames:
            # Pad with last frame if not enough
            while len(frames) < num_frames:
                frames.append(frames[-1])
        elif len(frames) > num_frames:
            # Sample evenly
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Preprocess frames
        processed_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)
        
        # Shape: (1, num_frames, 224, 224, 3)
        sequence = np.expand_dims(np.array(processed_frames), axis=0)
        
        # Get prediction
        prediction = self.lstm_model.predict(sequence, verbose=0)[0][0]
        
        return float(prediction)
    
    async def _get_real_predictions(self, image: np.ndarray) -> List[ModelPrediction]:
        """Get predictions from loaded models"""
        predictions = []
        
        # Xception prediction
        if self.xception_model is not None:
            xception_input = preprocess_for_xception(image)
            xception_pred = self.xception_model.predict(np.expand_dims(xception_input, 0), verbose=0)[0][0]
            predictions.append(ModelPrediction(
                model_name="Xception",
                fake_probability=float(xception_pred),
                confidence=abs(2 * xception_pred - 1),
                weight=settings.XCEPTION_WEIGHT
            ))
        
        # EfficientNet prediction
        if self.efficientnet_model is not None:
            effnet_input = preprocess_for_efficientnet(image)
            effnet_pred = self.efficientnet_model.predict(np.expand_dims(effnet_input, 0), verbose=0)[0][0]
            predictions.append(ModelPrediction(
                model_name="EfficientNet-B4",
                fake_probability=float(effnet_pred),
                confidence=abs(2 * effnet_pred - 1),
                weight=settings.EFFICIENTNET_WEIGHT
            ))
        
        # If no models loaded, fall back to simulated
        if not predictions:
            return self._get_simulated_predictions()
        
        return predictions
    
    def _get_simulated_predictions(self) -> List[ModelPrediction]:
        """Generate simulated predictions for demo mode"""
        # Generate realistic-looking predictions
        base_prob = random.uniform(0.2, 0.8)
        noise = 0.1
        
        predictions = [
            ModelPrediction(
                model_name="Xception",
                fake_probability=round(min(1, max(0, base_prob + random.uniform(-noise, noise))), 4),
                confidence=round(random.uniform(0.75, 0.95), 4),
                weight=settings.XCEPTION_WEIGHT
            ),
            ModelPrediction(
                model_name="EfficientNet-B4",
                fake_probability=round(min(1, max(0, base_prob + random.uniform(-noise, noise))), 4),
                confidence=round(random.uniform(0.78, 0.96), 4),
                weight=settings.EFFICIENTNET_WEIGHT
            ),
            ModelPrediction(
                model_name="CNN+LSTM",
                fake_probability=round(min(1, max(0, base_prob + random.uniform(-noise, noise))), 4),
                confidence=round(random.uniform(0.72, 0.92), 4),
                weight=settings.LSTM_WEIGHT
            )
        ]
        
        return predictions
    
    def _ensemble_vote(self, predictions: List[ModelPrediction]) -> float:
        """Calculate weighted ensemble prediction"""
        if not predictions:
            return 0.5
        
        weighted_sum = sum(p.fake_probability * p.weight for p in predictions)
        total_weight = sum(p.weight for p in predictions)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _calculate_confidence(self, predictions: List[ModelPrediction]) -> float:
        """Calculate overall confidence from model predictions"""
        if not predictions:
            return 0.5
        
        # Weighted average of confidences
        weighted_conf = sum(p.confidence * p.weight for p in predictions)
        total_weight = sum(p.weight for p in predictions)
        
        # Penalize disagreement between models
        probs = [p.fake_probability for p in predictions]
        variance = np.var(probs)
        disagreement_penalty = min(0.2, variance)
        
        base_confidence = weighted_conf / total_weight if total_weight > 0 else 0.5
        return max(0.1, base_confidence - disagreement_penalty)
    
    def _calculate_confidence_interval(self, predictions: List[ModelPrediction]) -> ConfidenceInterval:
        """Calculate 95% confidence interval for the prediction"""
        probs = [p.fake_probability for p in predictions]
        
        if len(probs) < 2:
            return ConfidenceInterval(
                lower=max(0, probs[0] - 0.1) if probs else 0.4,
                upper=min(1, probs[0] + 0.1) if probs else 0.6,
                confidence_level=0.95
            )
        
        mean = np.mean(probs)
        std = np.std(probs)
        
        # 95% CI = mean ± 1.96 * std
        return ConfidenceInterval(
            lower=round(max(0, mean - 1.96 * std), 4),
            upper=round(min(1, mean + 1.96 * std), 4),
            confidence_level=0.95
        )
    
    def _determine_risk_level(self, fake_probability: float) -> str:
        """Determine risk level based on fake probability"""
        if fake_probability >= 0.85:
            return "critical"
        elif fake_probability >= 0.65:
            return "high"
        elif fake_probability >= 0.35:
            return "medium"
        else:
            return "low"
    
    def _generate_notes(
        self, 
        fake_probability: float, 
        face_detected: bool,
        predictions: List[ModelPrediction]
    ) -> List[str]:
        """Generate contextual notes for the result"""
        notes = []
        
        # Simulation mode note
        if not self.models_loaded:
            notes.append("⚠️ Running in simulation mode - results are for demonstration only")
        
        # Face detection note
        if not face_detected:
            notes.append("No face detected in image - analysis performed on full image")
        
        # Uncertainty notes
        if 0.35 <= fake_probability <= 0.65:
            notes.append("Result is in uncertain range - additional verification recommended")
        
        # Model agreement note
        if predictions:
            probs = [p.fake_probability for p in predictions]
            if np.std(probs) > 0.15:
                notes.append("Models show significant disagreement - interpret with caution")
        
        # High confidence notes
        if fake_probability >= 0.85:
            notes.append("High probability of manipulation detected")
        elif fake_probability <= 0.15:
            notes.append("High probability of authentic media")
        
        return notes
