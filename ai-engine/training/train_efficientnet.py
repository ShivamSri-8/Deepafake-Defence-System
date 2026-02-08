"""
EfficientNet-B4 Model Training Script
For deepfake detection on images
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse


def create_efficientnet_model(input_shape=(380, 380, 3), num_classes=1):
    """Create EfficientNet-B4 model for binary classification"""
    
    # Load pre-trained EfficientNet-B4
    base_model = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model


def main(args):
    print("=" * 60)
    print("EfficientNet-B4 Deepfake Detection Training")
    print("=" * 60)
    
    # Create model
    print("\n📦 Creating EfficientNet-B4 model...")
    model, base_model = create_efficientnet_model()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Data generators
    print("\n📁 Setting up data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(380, 380),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        args.data_dir,
        target_size=(380, 380),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    
    # Callbacks
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
            patience=5,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train classification head
    print("\n🚀 Phase 1: Training classification head...")
    history1 = model.fit(
        train_generator,
        epochs=args.epochs // 2,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Phase 2: Fine-tune top layers
    print("\n🔧 Phase 2: Fine-tuning top layers...")
    
    # Unfreeze top layers
    for layer in base_model.layers[-50:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    
    # Re-compile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(
        train_generator,
        epochs=args.epochs // 2,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 deepfake detector")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--output-path", type=str,
                        default="models/weights/efficientnet_deepfake.h5",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Initial learning rate")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    main(args)
