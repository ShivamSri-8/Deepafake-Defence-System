"""
CNN+LSTM Model Training Script
For deepfake detection in videos using temporal analysis
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization,
    TimeDistributed, GlobalAveragePooling2D, Input
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse
import cv2
from glob import glob
from sklearn.model_selection import train_test_split


def extract_frames(video_path: str, num_frames: int = 20) -> np.ndarray:
    """Extract evenly spaced frames from a video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        # If video is too short, sample what we can
        indices = list(range(total_frames))
    else:
        # Sample evenly spaced frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize to EfficientNet input size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    cap.release()
    
    # Pad if we don't have enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    return np.array(frames[:num_frames])


def create_cnn_lstm_model(num_frames: int = 20, input_shape: tuple = (224, 224, 3)):
    """
    Create CNN+LSTM model for video deepfake detection.
    Uses EfficientNetB0 as feature extractor + LSTM for temporal analysis.
    """
    # Input: sequence of frames
    input_layer = Input(shape=(num_frames, *input_shape))
    
    # CNN Feature Extractor (shared across time steps)
    cnn_base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze CNN base initially
    for layer in cnn_base.layers:
        layer.trainable = False
    
    # TimeDistributed CNN + Global Pooling
    x = TimeDistributed(cnn_base)(input_layer)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    
    # LSTM for temporal analysis
    x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
    x = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    
    # Classification head
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=predictions)
    
    return model, cnn_base


class VideoDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for video sequences"""
    
    def __init__(self, video_paths, labels, batch_size=4, num_frames=20, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle
        self.indices = np.arange(len(video_paths))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = []
        batch_y = []
        
        for i in batch_indices:
            frames = extract_frames(self.video_paths[i], self.num_frames)
            if frames is not None:
                # Normalize to [0, 1]
                frames = frames.astype(np.float32) / 255.0
                batch_x.append(frames)
                batch_y.append(self.labels[i])
        
        return np.array(batch_x), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_dataset(data_dir: str):
    """
    Load video dataset from directory structure:
    data_dir/
        real/
            video1.mp4
            video2.mp4
        fake/
            video1.mp4
            video2.mp4
    """
    real_videos = glob(os.path.join(data_dir, "real", "*.mp4"))
    fake_videos = glob(os.path.join(data_dir, "fake", "*.mp4"))
    
    # Also check for avi and mov files
    for ext in ["*.avi", "*.mov", "*.mkv"]:
        real_videos.extend(glob(os.path.join(data_dir, "real", ext)))
        fake_videos.extend(glob(os.path.join(data_dir, "fake", ext)))
    
    video_paths = real_videos + fake_videos
    labels = [0] * len(real_videos) + [1] * len(fake_videos)
    
    return video_paths, labels


def main(args):
    print("=" * 60)
    print("CNN+LSTM Deepfake Detection Training (Video)")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📁 Loading dataset from: {args.data_dir}")
    video_paths, labels = load_dataset(args.data_dir)
    
    if len(video_paths) == 0:
        print("❌ No videos found! Expected structure:")
        print("   data_dir/real/*.mp4")
        print("   data_dir/fake/*.mp4")
        return
    
    print(f"✅ Found {len(video_paths)} videos")
    print(f"   Real: {labels.count(0)}, Fake: {labels.count(1)}")
    
    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"✅ Training samples: {len(train_paths)}")
    print(f"✅ Validation samples: {len(val_paths)}")
    
    # Create data generators
    train_gen = VideoDataGenerator(
        train_paths, train_labels,
        batch_size=args.batch_size,
        num_frames=args.num_frames
    )
    
    val_gen = VideoDataGenerator(
        val_paths, val_labels,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        shuffle=False
    )
    
    # Create model
    print(f"\n📦 Creating CNN+LSTM model...")
    print(f"   Sequence length: {args.num_frames} frames")
    
    model, cnn_base = create_cnn_lstm_model(num_frames=args.num_frames)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Callbacks
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            args.output_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=7,
            mode='max',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train LSTM layers only (CNN frozen)
    print("\n🚀 Phase 1: Training LSTM layers (CNN frozen)...")
    history1 = model.fit(
        train_gen,
        epochs=args.epochs // 2,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune CNN + LSTM
    print("\n🔧 Phase 2: Fine-tuning entire model...")
    
    # Unfreeze top layers of CNN
    for layer in cnn_base.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    # Re-compile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(
        train_gen,
        epochs=args.epochs // 2,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    val_loss, val_acc, val_auc = model.evaluate(val_gen, verbose=0)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f}")
    print(f"   Validation AUC: {val_auc:.4f}")
    
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+LSTM deepfake detector for videos")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to video dataset directory (with real/ and fake/ subdirs)")
    parser.add_argument("--output-path", type=str,
                        default="models/weights/cnn_lstm_deepfake.h5",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (smaller for video due to memory)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--num-frames", type=int, default=20,
                        help="Number of frames to sample from each video")
    
    args = parser.parse_args()
    main(args)
