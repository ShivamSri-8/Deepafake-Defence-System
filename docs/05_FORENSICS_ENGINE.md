# Forensic Analysis Engine

---

## 1. Overview

The Forensic Analysis Engine provides multi-faceted analysis beyond deep learning classification, examining visual artifacts and inconsistencies that indicate manipulation.

---

## 2. Forensic Features Matrix

| Feature | Method | Deepfake Indicators |
|---------|--------|---------------------|
| **Facial Landmarks** | MediaPipe + Dlib (468 points) | Asymmetric movements, unnatural positions |
| **Eye Blink Analysis** | Eye Aspect Ratio (EAR) | Abnormal blink frequency, missing blinks |
| **Lip Sync Coherence** | Audio-Visual Correlation | Mouth movement vs audio mismatch |
| **Frequency Analysis** | 2D FFT + DCT | GAN fingerprints in frequency domain |
| **Temporal Consistency** | Optical Flow + Frame Diff | Flickering, unnatural transitions |
| **Pixel-Level Artifacts** | Local Binary Patterns | Compression artifacts, blending borders |

---

## 3. Forensic Pipeline Architecture

```
Input: Media (Image/Video)
    │
    ├──▶ Face Detection (RetinaFace/MTCNN)
    │       └── Extract face ROI
    │
    ├──▶ Landmark Analysis (468 points)
    │       └── Compute symmetry scores
    │       └── Track movement trajectories
    │
    ├──▶ Eye Region Analysis
    │       └── Detect blinks (EAR threshold)
    │       └── Compute blink rate statistics
    │
    ├──▶ Frequency Domain Analysis
    │       └── 2D FFT on face region
    │       └── Extract GAN artifact signatures
    │
    ├──▶ Temporal Analysis (Video only)
    │       └── Optical flow computation
    │       └── Frame-to-frame consistency
    │
    └──▶ Aggregate Forensic Report
            └── Feature scores
            └── Anomaly flags
            └── Visual overlays
```

---

## 4. Implementation Details

### 4.1 Face Analyzer

```python
import cv2
import mediapipe as mp
import numpy as np

class FaceAnalyzer:
    """
    Analyzes facial landmarks for deepfake detection.
    """
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def analyze(self, frame):
        """Analyze frame for facial landmark anomalies."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {'detected': False}
        
        landmarks = results.multi_face_landmarks[0]
        
        # Extract key measurements
        symmetry_score = self._compute_symmetry(landmarks)
        movement_smoothness = self._analyze_movement(landmarks)
        
        return {
            'detected': True,
            'symmetry_score': symmetry_score,
            'anomaly_detected': symmetry_score < 0.85,
            'landmark_count': len(landmarks.landmark),
            'movement_smoothness': movement_smoothness
        }
    
    def _compute_symmetry(self, landmarks):
        """Compute facial symmetry score (0-1)."""
        # Extract left and right side landmarks
        left_eye = self._get_landmark_coords(landmarks, [33, 133, 160, 158])
        right_eye = self._get_landmark_coords(landmarks, [362, 263, 387, 385])
        
        # Compute symmetry between sides
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Normalize and compute similarity
        symmetry = 1.0 - abs(left_center[1] - right_center[1])
        return max(0, min(1, symmetry))
```

### 4.2 Blink Detector

```python
class BlinkDetector:
    """
    Detects and analyzes eye blink patterns.
    """
    
    EAR_THRESHOLD = 0.21
    CONSECUTIVE_FRAMES = 3
    
    # Eye landmark indices (MediaPipe)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def __init__(self):
        self.frame_counter = 0
        self.blink_counter = 0
        self.ear_history = []
    
    def compute_ear(self, landmarks, eye_indices):
        """
        Compute Eye Aspect Ratio (EAR).
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        coords = np.array([
            [landmarks.landmark[i].x, landmarks.landmark[i].y]
            for i in eye_indices
        ])
        
        # Vertical distances
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        
        # Horizontal distance
        h = np.linalg.norm(coords[0] - coords[3])
        
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def analyze_blinks(self, video_frames, landmarks_list):
        """Analyze blink patterns across video frames."""
        ear_values = []
        
        for landmarks in landmarks_list:
            left_ear = self.compute_ear(landmarks, self.LEFT_EYE)
            right_ear = self.compute_ear(landmarks, self.RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)
        
        # Detect blinks
        blink_frames = []
        for i, ear in enumerate(ear_values):
            if ear < self.EAR_THRESHOLD:
                blink_frames.append(i)
        
        # Compute statistics
        total_frames = len(video_frames)
        blink_rate = len(blink_frames) / (total_frames / 30.0)  # per second
        
        # Normal blink rate: 15-20 per minute (0.25-0.33 per second)
        normal_range = (0.2, 0.5)
        anomaly = blink_rate < normal_range[0] or blink_rate > normal_range[1]
        
        return {
            'blink_count': len(blink_frames),
            'blink_rate_per_second': blink_rate,
            'ear_values': ear_values,
            'anomaly_detected': anomaly,
            'blink_frames': blink_frames
        }
```

### 4.3 Frequency Analyzer

```python
class FrequencyAnalyzer:
    """
    Analyzes frequency domain for GAN artifacts.
    """
    
    def analyze(self, face_image):
        """
        Detect GAN fingerprints in frequency domain.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Log transform for visualization
        magnitude_db = 20 * np.log10(magnitude_spectrum + 1)
        
        # Analyze high-frequency components
        h, w = magnitude_db.shape
        center_h, center_w = h // 2, w // 2
        
        # Extract radial profile
        radial_profile = self._compute_radial_profile(magnitude_db)
        
        # GAN artifacts typically show periodic patterns
        artifact_score = self._detect_periodic_artifacts(radial_profile)
        
        return {
            'magnitude_spectrum': magnitude_db,
            'artifact_score': artifact_score,
            'gan_fingerprint_detected': artifact_score > 0.6,
            'radial_profile': radial_profile
        }
    
    def _compute_radial_profile(self, spectrum):
        """Compute radially averaged frequency profile."""
        h, w = spectrum.shape
        center = (h // 2, w // 2)
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # Average by radius
        max_r = min(center)
        profile = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            profile[i] = spectrum[mask].mean()
        
        return profile
    
    def _detect_periodic_artifacts(self, profile):
        """Detect periodic patterns indicative of GAN generation."""
        # Use autocorrelation to find periodicity
        acf = np.correlate(profile, profile, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        
        # Find significant peaks
        peaks = []
        for i in range(5, len(acf) - 5):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.3:
                peaks.append(i)
        
        # More peaks = more likely GAN artifact
        score = min(1.0, len(peaks) / 5.0)
        return score
```

### 4.4 Temporal Consistency Analyzer

```python
class TemporalAnalyzer:
    """
    Analyzes temporal consistency across video frames.
    """
    
    def analyze(self, frames):
        """
        Analyze frame-to-frame consistency.
        """
        if len(frames) < 2:
            return {'error': 'Insufficient frames'}
        
        flow_scores = []
        diff_scores = []
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_scores.append(np.mean(flow_magnitude))
            
            # Frame difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            diff_scores.append(np.mean(diff))
        
        # Analyze variance - inconsistent = likely manipulated
        flow_variance = np.var(flow_scores)
        diff_variance = np.var(diff_scores)
        
        # Detect flickering (high variance in differences)
        flicker_detected = diff_variance > 100
        
        # Detect unnatural motion (abnormal flow patterns)
        motion_anomaly = flow_variance > np.mean(flow_scores) * 0.5
        
        return {
            'flow_scores': flow_scores,
            'diff_scores': diff_scores,
            'flow_variance': flow_variance,
            'diff_variance': diff_variance,
            'flicker_detected': flicker_detected,
            'motion_anomaly': motion_anomaly,
            'temporal_score': 1.0 - min(1.0, flow_variance / 10.0)
        }
```

---

## 5. Aggregated Forensic Report

```python
class ForensicReport:
    """
    Aggregates all forensic analyses into a unified report.
    """
    
    def generate(self, face_analysis, blink_analysis, 
                 frequency_analysis, temporal_analysis=None):
        
        indicators = []
        scores = {}
        
        # Face analysis
        if face_analysis.get('anomaly_detected'):
            indicators.append({
                'feature': 'facial_landmarks',
                'status': 'anomaly',
                'score': 1 - face_analysis['symmetry_score']
            })
        scores['facial_landmarks'] = face_analysis.get('symmetry_score', 0)
        
        # Blink analysis
        if blink_analysis.get('anomaly_detected'):
            indicators.append({
                'feature': 'eye_blink',
                'status': 'anomaly',
                'score': 0.8
            })
        scores['eye_blink'] = 0.2 if blink_analysis.get('anomaly_detected') else 0.8
        
        # Frequency analysis
        if frequency_analysis.get('gan_fingerprint_detected'):
            indicators.append({
                'feature': 'frequency_domain',
                'status': 'anomaly',
                'score': frequency_analysis['artifact_score']
            })
        scores['frequency'] = 1 - frequency_analysis.get('artifact_score', 0)
        
        # Temporal analysis (video only)
        if temporal_analysis:
            if temporal_analysis.get('flicker_detected'):
                indicators.append({
                    'feature': 'temporal_consistency',
                    'status': 'anomaly',
                    'score': 0.7
                })
            scores['temporal'] = temporal_analysis.get('temporal_score', 0)
        
        # Overall forensic score
        overall_score = np.mean(list(scores.values()))
        
        return {
            'overall_score': overall_score,
            'indicators': indicators,
            'detailed_scores': scores,
            'recommendation': 'likely_fake' if overall_score < 0.5 else 'likely_real'
        }
```

---

*Document Version: 1.0 | Created: 2026-02-07*
