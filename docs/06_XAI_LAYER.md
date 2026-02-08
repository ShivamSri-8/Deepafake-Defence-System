# Explainable AI (XAI) Layer

---

## 1. Overview

The XAI layer ensures every prediction is accompanied by interpretable explanations, making the system transparent and trustworthy for academic and professional use.

---

## 2. Explainability Methods

| Method | Purpose | Output |
|--------|---------|--------|
| **Grad-CAM** | Visual attention maps | Heatmap overlay on input |
| **Grad-CAM++** | Improved localization | Enhanced heatmap |
| **LIME** | Local interpretable explanations | Superpixel importance |
| **SHAP** | Feature attribution | Contribution scores |

---

## 3. Grad-CAM Implementation

```python
import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visual explanations.
    """
    
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
        # Create gradient model
        self.grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[
                model.get_layer(self.layer_name).output,
                model.output
            ]
        )
    
    def _find_target_layer(self):
        """Find the last convolutional layer."""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # Conv layer
                return layer.name
        raise ValueError("No convolutional layer found")
    
    def compute_heatmap(self, image, class_idx=None):
        """
        Generate Grad-CAM heatmap for the input image.
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        # Gradient of output w.r.t. conv layer
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize to [0, 1]
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image.
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
    
    def identify_key_regions(self, heatmap, threshold=0.5):
        """
        Identify key regions contributing to prediction.
        """
        # Threshold heatmap
        binary = (heatmap > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            attention = np.mean(heatmap[y:y+h, x:x+w])
            regions.append({
                'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                'attention_score': float(attention),
                'area': w * h
            })
        
        # Sort by attention score
        regions.sort(key=lambda r: r['attention_score'], reverse=True)
        return regions[:5]  # Top 5 regions
```

---

## 4. LIME Implementation

```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations.
    """
    
    def __init__(self, model, preprocess_fn=None):
        self.model = model
        self.preprocess_fn = preprocess_fn or (lambda x: x / 255.0)
        self.explainer = lime_image.LimeImageExplainer()
    
    def explain(self, image, num_samples=1000, num_features=10):
        """
        Generate LIME explanation for the image.
        """
        def predict_fn(images):
            processed = np.array([self.preprocess_fn(img) for img in images])
            return self.model.predict(processed)
        
        explanation = self.explainer.explain_instance(
            image,
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation
    
    def visualize(self, image, explanation, positive_only=True):
        """
        Create visualization of LIME explanation.
        """
        # Get mask for top features
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=positive_only,
            num_features=10,
            hide_rest=False
        )
        
        # Create marked image
        marked = mark_boundaries(temp / 255.0, mask)
        return (marked * 255).astype(np.uint8)
    
    def get_feature_importance(self, explanation):
        """
        Extract feature importance scores.
        """
        local_exp = explanation.local_exp[explanation.top_labels[0]]
        
        features = []
        for segment_id, importance in local_exp:
            features.append({
                'segment_id': segment_id,
                'importance': float(importance),
                'direction': 'positive' if importance > 0 else 'negative'
            })
        
        return sorted(features, key=lambda x: abs(x['importance']), reverse=True)
```

---

## 5. Text Explanation Generator

```python
class ExplanationGenerator:
    """
    Generates human-readable text explanations for predictions.
    """
    
    REGION_NAMES = {
        'upper_left': 'forehead/hair region',
        'upper_right': 'forehead/hair region',
        'center_left': 'left eye area',
        'center': 'nose/central face',
        'center_right': 'right eye area',
        'lower_left': 'left jaw/cheek',
        'lower_center': 'mouth/lip region',
        'lower_right': 'right jaw/cheek'
    }
    
    def generate(self, prediction, gradcam_regions, forensic_report):
        """
        Generate comprehensive text explanation.
        """
        explanation_parts = []
        
        # Main prediction statement
        prob = prediction['probability']
        conf_lower = prediction['confidence_lower']
        conf_upper = prediction['confidence_upper']
        
        if prob > 0.7:
            explanation_parts.append(
                f"The model detected potential manipulation indicators with "
                f"{prob*100:.1f}% probability (95% CI: {conf_lower*100:.1f}%-{conf_upper*100:.1f}%)."
            )
        elif prob > 0.4:
            explanation_parts.append(
                f"The model found uncertain indicators with {prob*100:.1f}% probability. "
                f"This result requires additional verification."
            )
        else:
            explanation_parts.append(
                f"The model found limited manipulation indicators ({prob*100:.1f}% probability). "
                f"The media appears likely authentic, but certainty is not guaranteed."
            )
        
        # Grad-CAM region insights
        if gradcam_regions:
            top_region = gradcam_regions[0]
            region_name = self._get_region_name(top_region['bbox'])
            explanation_parts.append(
                f"The {region_name} showed the highest attention "
                f"(score: {top_region['attention_score']:.2f}), indicating "
                f"this area most influenced the model's decision."
            )
        
        # Forensic insights
        if forensic_report:
            for indicator in forensic_report.get('indicators', []):
                feature = indicator['feature'].replace('_', ' ')
                if indicator['status'] == 'anomaly':
                    explanation_parts.append(
                        f"The {feature} analysis detected anomalies "
                        f"(confidence: {indicator['score']*100:.0f}%)."
                    )
        
        # Compile explanations
        return {
            'summary': ' '.join(explanation_parts),
            'detailed': explanation_parts,
            'confidence_level': self._get_confidence_level(prob),
            'disclaimer': self._get_disclaimer()
        }
    
    def _get_region_name(self, bbox):
        """Map bounding box to human-readable region name."""
        # Assuming normalized coordinates
        cx = bbox['x'] + bbox['w'] / 2
        cy = bbox['y'] + bbox['h'] / 2
        
        if cy < 0.33:
            vert = 'upper'
        elif cy < 0.66:
            vert = 'center'
        else:
            vert = 'lower'
        
        if cx < 0.33:
            horiz = 'left'
        elif cx < 0.66:
            horiz = 'center'
        else:
            horiz = 'right'
        
        key = f"{vert}_{horiz}"
        return self.REGION_NAMES.get(key, 'facial region')
    
    def _get_confidence_level(self, prob):
        """Categorize confidence level."""
        if prob > 0.85 or prob < 0.15:
            return 'high'
        elif prob > 0.7 or prob < 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def _get_disclaimer(self):
        """Return standard disclaimer text."""
        return (
            "This is a probabilistic assessment by an AI system. "
            "Results should not be considered definitive proof. "
            "Verification by qualified experts is recommended."
        )
```

---

## 6. Complete Explainability Engine

```python
class ExplainabilityEngine:
    """
    Unified explainability engine combining all XAI methods.
    """
    
    def __init__(self, model):
        self.model = model
        self.gradcam = GradCAM(model)
        self.lime = LIMEExplainer(model)
        self.text_generator = ExplanationGenerator()
    
    def explain(self, image, prediction, forensic_report=None):
        """
        Generate comprehensive explanation for a prediction.
        """
        # Grad-CAM analysis
        heatmap = self.gradcam.compute_heatmap(image)
        overlay = self.gradcam.overlay_heatmap(image, heatmap)
        key_regions = self.gradcam.identify_key_regions(heatmap)
        
        # LIME analysis (optional, computationally expensive)
        lime_explanation = self.lime.explain(image, num_samples=500)
        lime_viz = self.lime.visualize(image, lime_explanation)
        lime_features = self.lime.get_feature_importance(lime_explanation)
        
        # Generate text explanation
        text_explanation = self.text_generator.generate(
            prediction, key_regions, forensic_report
        )
        
        return {
            'visual': {
                'gradcam_heatmap': heatmap,
                'gradcam_overlay': overlay,
                'lime_visualization': lime_viz,
                'key_regions': key_regions
            },
            'textual': text_explanation,
            'feature_importance': lime_features[:10],
            'model_attention': {
                'top_region': key_regions[0] if key_regions else None,
                'attention_distribution': self._compute_attention_distribution(heatmap)
            }
        }
    
    def _compute_attention_distribution(self, heatmap):
        """Compute attention distribution across image quadrants."""
        h, w = heatmap.shape
        quadrants = {
            'top_left': heatmap[:h//2, :w//2],
            'top_right': heatmap[:h//2, w//2:],
            'bottom_left': heatmap[h//2:, :w//2],
            'bottom_right': heatmap[h//2:, w//2:]
        }
        
        distribution = {}
        for name, region in quadrants.items():
            distribution[name] = float(np.mean(region))
        
        return distribution
```

---

## 7. Visualization Output Example

```
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION RESULT                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CLASSIFICATION: POTENTIALLY MANIPULATED                       │
│   CONFIDENCE: 78.3% (±5.2%)                                     │
│                                                                  │
│   ┌──────────────────┐   ┌──────────────────────────────────┐   │
│   │                  │   │  KEY OBSERVATIONS:                │   │
│   │   [Grad-CAM      │   │                                   │   │
│   │    Heatmap       │   │  • Face boundary shows blending   │   │
│   │    Overlay]      │   │    artifacts (attention: 0.89)    │   │
│   │                  │   │                                   │   │
│   │  🔴 High Attn    │   │  • Eye region has temporal        │   │
│   │  🟡 Medium Attn  │   │    inconsistencies                │   │
│   │  🟢 Low Attn     │   │                                   │   │
│   └──────────────────┘   │  • Lip sync coherence: 67%        │   │
│                          └──────────────────────────────────┘   │
│                                                                  │
│   ⚠️ DISCLAIMER: This is a probabilistic analysis. Results     │
│   should be verified by domain experts.                         │
└─────────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.0 | Created: 2026-02-07*
