"""
Deepfake Detection Service — EDDS AI Engine
Production Inference Pipeline:
  • Single Model: PyTorch EfficientNet-B4
  • Frame batching for memory optimization
  • Hard failure on missing model (no simulation)
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from typing import List

from config import settings
from models.schemas import (
    DetectionResult,
    ModelPrediction,
    ConfidenceInterval,
)
from utils.preprocessing import (
    load_image,
    extract_face,
    extract_video_frames,
)
from utils.logger import setup_logger
from utils.trust_score import (
    compute_agreement_score,
    compute_trust_score,
    compute_temporal_variance,
    compute_confidence_level,
)

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch DataLoader Helpers
# ─────────────────────────────────────────────────────────────────────────────
class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, frames_np, transform=None):
        self.frames = frames_np
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        if self.transform:
            return self.transform(img)
        return img


def get_inference_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((settings.EFFICIENTNET_SIZE[0], settings.EFFICIENTNET_SIZE[1])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Detector class
# ─────────────────────────────────────────────────────────────────────────────
class DeepfakeDetector:
    """
    Production Deepfake Detector using PyTorch EfficientNet-B4.
    """

    def __init__(self):
        self.models_loaded = False
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_inference_transform()
        
        self._load_models()

    def _load_models(self):
        logger.info(f"Starting Production Inference Engine on {self.device}...")
        
        if not os.path.exists(settings.MODEL_PATH):
            error_msg = f"❌ CRITICAL ERROR: Model weights not found at {settings.MODEL_PATH}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            logger.info("Initializing EfficientNet-B4 architecture...")
            # Initialize model architecture
            self.model = tv_models.efficientnet_b4()
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
            
            # Load weights
            self.model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            
            self.models_loaded = True
            logger.info(f"✅ Loaded production model successfully: {settings.MODEL_PATH}")
            
        except Exception as e:
            error_msg = f"❌ Failed to load model architecture or weights: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    async def detect_image(self, image_path: str) -> DetectionResult:
        """Detect deepfake in a single image."""
        logger.info(f"Analysing image: {image_path}")

        image = load_image(image_path)
        face_result = extract_face(image)
        face_detected = face_result is not None
        face_image = face_result[0] if face_result else image

        if face_detected:
            logger.info("Face detected ✅")
        else:
            logger.warning("No face detected — using full image")

        # Run inference
        prob = self._infer_single(face_image)
        
        pred = ModelPrediction(
            model_name="EfficientNet-B4 (Production)",
            fake_probability=round(prob, 4),
            confidence=round(abs(2 * prob - 1), 4),
            weight=1.0,
            is_simulated=False
        )

        fake_probability = prob
        ci = self._confidence_interval([pred])
        risk = self._risk_level(fake_probability)
        confidence_val = pred.confidence

        # ── Trust-Aware Fields ────────────────────────────────────
        trust = compute_trust_score(confidence_val, 1.0) # Agreement is 1.0 for single model
        conf_lvl = compute_confidence_level(confidence_val * 100)

        return DetectionResult(
            is_fake=fake_probability >= settings.FAKE_THRESHOLD,
            fake_probability=round(fake_probability, 4),
            confidence=round(confidence_val, 4),
            confidence_interval=ci,
            risk_level=risk,
            model_predictions=[pred],
            face_detected=face_detected,
            notes=self._build_notes(fake_probability, face_detected),
            trust_score=trust,
            temporal_variance=None,
            temporal_label=None,
            confidence_level=conf_lvl,
        )

    async def detect_video(self, video_path: str) -> DetectionResult:
        """Detect deepfake in a video by analysing sampled frames using batching."""
        logger.info(f"Analysing video: {video_path}")

        # Extract frames directly
        frames = extract_video_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted from video")

        logger.info(f"Extracted {len(frames)} frames. Running batched inference...")

        # Process faces for each frame
        face_frames = []
        for frame in frames:
            face_res = extract_face(frame)
            face_img = face_res[0] if face_res else frame
            face_frames.append(face_img)

        # Batch Inference to save memory
        frame_scores = self._infer_batch(face_frames)

        mean_score = float(np.mean(frame_scores))
        std_score  = float(np.std(frame_scores))

        # ── Frame-level deepfake count ─────────────────────────────────────
        deepfake_frames  = sum(1 for s in frame_scores if s >= settings.FAKE_THRESHOLD)
        authentic_frames = len(frame_scores) - deepfake_frames

        frame_pred = ModelPrediction(
            model_name="Video Frame Analysis (EfficientNet)",
            fake_probability=round(mean_score, 4),
            confidence=round(max(0.1, 1 - std_score), 4),
            weight=1.0,
            is_simulated=False
        )

        notes = [
            f"Frames analysed: {len(frame_scores)}",
            f"Deepfake frames: {deepfake_frames} / {len(frame_scores)}",
            f"Authentic frames: {authentic_frames} / {len(frame_scores)}",
            f"Authenticity score: {1 - mean_score:.2f}",
            f"Frame-to-frame variance: {std_score:.4f}",
        ]

        if std_score > 0.20:
            notes.append("⚠️ High frame variance — possible manipulation boundary")

        fake_probability = mean_score
        risk = self._risk_level(fake_probability)
        confidence_val = max(0.1, 1 - std_score)

        ci = ConfidenceInterval(
            lower=round(max(0.0, mean_score - 1.96 * std_score), 4),
            upper=round(min(1.0, mean_score + 1.96 * std_score), 4),
            confidence_level=0.95,
        )

        # ── Trust-Aware Fields ────────────────────────────────────
        agreement = compute_agreement_score(frame_scores)
        trust = compute_trust_score(confidence_val, agreement)
        t_var, t_label = compute_temporal_variance(frame_scores)
        conf_lvl = compute_confidence_level(confidence_val * 100, t_var)

        return DetectionResult(
            is_fake=fake_probability >= settings.FAKE_THRESHOLD,
            fake_probability=round(fake_probability, 4),
            confidence=round(confidence_val, 4),
            confidence_interval=ci,
            risk_level=risk,
            model_predictions=[frame_pred],
            face_detected=True,
            notes=notes,
            trust_score=trust,
            temporal_variance=t_var,
            temporal_label=t_label,
            confidence_level=conf_lvl,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Inference helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _infer_single(self, image_np: np.ndarray) -> float:
        """Run single image through the model."""
        tensor = self.transform(image_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.sigmoid(output).item()
        return float(prob)
        
    def _infer_batch(self, images: List[np.ndarray]) -> List[float]:
        """Run a batch of images through the model efficiently."""
        dataset = FrameDataset(images, transform=self.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=settings.BATCH_SIZE, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch_tensors in dataloader:
                batch_tensors = batch_tensors.to(self.device)
                outputs = self.model(batch_tensors)
                probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
                if probs.ndim == 0:
                    probs = [float(probs)]
                all_probs.extend([float(p) for p in probs])
                
        return all_probs

    def _confidence_interval(self, preds: List[ModelPrediction]) -> ConfidenceInterval:
        probs = [p.fake_probability for p in preds]
        v = probs[0] if probs else 0.5
        return ConfidenceInterval(lower=max(0, v - 0.1), upper=min(1, v + 0.1), confidence_level=0.95)

    def _risk_level(self, p: float) -> str:
        if p >= 0.85:  return "critical"
        if p >= settings.FAKE_THRESHOLD:  return "high"
        if p >= settings.SUSPICIOUS_THRESHOLD:  return "medium"
        return "low"

    def _build_notes(self, p: float, face_detected: bool) -> List[str]:
        notes = []
        if not face_detected:
            notes.append("No face detected — analysis performed on full image")
        if settings.SUSPICIOUS_THRESHOLD <= p < settings.FAKE_THRESHOLD:
            notes.append("Result is in the suspicious range — additional verification recommended")
        elif p >= settings.FAKE_THRESHOLD:
            notes.append(f"High probability of manipulation detected ({p:.0%})")
        elif p < settings.LOW_CONFIDENCE_THRESHOLD:
            notes.append(f"Media appears likely authentic ({(1-p):.0%} confidence)")
        return notes
