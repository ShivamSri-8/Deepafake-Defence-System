"""
5-PHASE MODULAR TRAINING PLAN FOR DEEPFAKE DETECTION
Divides training into manageable phases for efficiency and faster iteration
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import argparse
from datetime import datetime

# ============================================================================
# PHASE 1: DATA VALIDATION & PREPARATION
# ============================================================================
def phase_1_validate_data(data_root="../data/140k_extracted/real_vs_fake/real-vs-fake"):
    """
    PHASE 1: Quick data validation
    - Check if all images are accessible
    - Verify directory structure
    - Create data inventory
    Duration: ~2-3 minutes
    """
    print("\n" + "="*70)
    print("PHASE 1: DATA VALIDATION & PREPARATION")
    print("="*70)
    
    data_path = Path(data_root)
    inventory = {}
    
    for split in ['train', 'valid', 'test']:
        split_path = data_path / split
        inventory[split] = {}
        
        for label in ['fake', 'real']:
            label_path = split_path / label
            if label_path.exists():
                count = len(list(label_path.glob("*.jpg")))
                inventory[split][label] = count
                print(f"  {split}/{label}: {count:,} images")
            else:
                print(f"  WARNING: {split}/{label} not found!")
                return False
    
    # Save inventory
    with open("data_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)
    
    print("\n✓ Data validation COMPLETE - Inventory saved to data_inventory.json")
    return True


# ============================================================================
# PHASE 2: MODEL ARCHITECTURE & TRANSFER LEARNING SETUP
# ============================================================================
def phase_2_setup_model(input_shape=(299, 299, 3), num_classes=1):
    """
    PHASE 2: Create model architecture with transfer learning
    - Load pre-trained Xception
    - Add custom classification head
    - Freeze base model layers
    Duration: ~1-2 minutes
    """
    print("\n" + "="*70)
    print("PHASE 2: MODEL ARCHITECTURE SETUP")
    print("="*70)
    
    print(f"\n  Loading pre-trained Xception...")
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Build classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"\n  Model Parameters: {model.count_params():,}")
    print(f"  Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print("\n✓ Model setup COMPLETE")
    
    return model, base_model


# ============================================================================
# PHASE 3: QUICK TRAINING WITH SMALL DATASET (Sanity Check)
# ============================================================================
def phase_3_quick_train(model, epochs=3, batch_size=32, data_dir="../data/images_small"):
    """
    PHASE 3: Quick training with small dataset subset
    - Train on 128x128 small images (1000 images total)
    - Quick validation (3 epochs)
    - Verify pipeline works
    Duration: ~5-10 minutes (GPU), ~30-40 minutes (CPU)
    """
    print("\n" + "="*70)
    print("PHASE 3: QUICK TRAINING (Small Dataset Sanity Check)")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    
    train_generator = train_datagen.flow_from_directory(
        str(data_path),
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    print(f"\n  Training on {len(train_generator.samples)} small images")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=len(train_generator),
        verbose=1
    )
    
    # Save quick checkpoint
    model.save("checkpoints/phase_3_quick_train.h5")
    
    print("\n✓ Quick training COMPLETE - Checkpoint saved")
    return history


# ============================================================================
# PHASE 4: FULL TRAINING PART 1 (Head Training)
# ============================================================================
def phase_4_train_head(model, epochs=10, batch_size=32, 
                       data_dir="../data/140k_extracted/real_vs_fake/real-vs-fake"):
    """
    PHASE 4: Full dataset training - Stage 1 (Classification head only)
    - Train only the new classification layers
    - Base Xception remains frozen
    - 100k training images
    Duration: ~2-4 hours (GPU), 1-2 days (CPU)
    """
    print("\n" + "="*70)
    print("PHASE 4: FULL TRAINING - STAGE 1 (Classification Head)")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1
    )
    
    train_generator = train_datagen.flow_from_directory(
        str(data_path / "train"),
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        str(data_path / "valid"),
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"\n  Training samples: {train_generator.samples:,}")
    print(f"  Validation samples: {val_generator.samples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            "checkpoints/phase_4_best_head.h5",
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=3,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save("checkpoints/phase_4_head_trained.h5")
    
    print("\n✓ Head training COMPLETE - Checkpoint saved")
    return history


# ============================================================================
# PHASE 5: FINE-TUNING (Unfreeze & Train Full Model)
# ============================================================================
def phase_5_fine_tune(model, base_model, epochs=15, batch_size=16,
                      data_dir="../data/140k_extracted/real_vs_fake/real-vs-fake",
                      unfreeze_layers=50):
    """
    PHASE 5: Full model fine-tuning
    - Unfreeze last 50 layers of Xception
    - Lower learning rate (0.0001)
    - Full end-to-end training
    Duration: ~4-8 hours (GPU), 2-3 days (CPU)
    """
    print("\n" + "="*70)
    print("PHASE 5: FINE-TUNING (Full Model)")
    print("="*70)
    
    # Unfreeze last N layers
    print(f"\n  Unfreezing last {unfreeze_layers} layers...")
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print(f"  Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    data_path = Path(data_dir)
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1
    )
    
    train_generator = train_datagen.flow_from_directory(
        str(data_path / "train"),
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        str(data_path / "valid"),
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_datagen.flow_from_directory(
        str(data_path / "test"),
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"\n  Training samples: {train_generator.samples:,}")
    print(f"  Validation samples: {val_generator.samples:,}")
    print(f"  Test samples: {test_generator.samples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            "checkpoints/phase_5_best_finetuned.h5",
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n\nEvaluating on test set...")
    test_loss, test_acc, test_auc = model.evaluate(test_generator)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    
    # Save final model
    model.save("xception_deepfake_final.h5")
    
    print("\n✓ Fine-tuning COMPLETE - Final model saved to xception_deepfake_final.h5")
    return history


# ============================================================================
# MAIN TRAINING ORCHESTRATOR
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="5-Phase Training Plan")
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4, 5, 0], 
                       default=0, help='Which phase to run (0=all, 1-5=specific)')
    parser.add_argument('--data-dir', default='../data/140k_extracted/real_vs_fake/real-vs-fake',
                       help='Path to training data')
    args = parser.parse_args()
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION: 5-PHASE TRAINING ORCHESTRATOR")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Validation
    if args.phase in [0, 1]:
        if not phase_1_validate_data(args.data_dir):
            print("ERROR: Data validation failed!")
            return
    
    # Phase 2: Setup
    if args.phase in [0, 2, 3, 4, 5]:
        model, base_model = phase_2_setup_model()
    
    # Phase 3: Quick training
    if args.phase in [0, 3]:
        phase_3_quick_train(model)
    
    # Phase 4: Full training (head)
    if args.phase in [0, 4]:
        phase_4_train_head(model)
    
    # Phase 5: Fine-tuning
    if args.phase in [0, 5]:
        phase_5_fine_tune(model, base_model)
    
    print("\n" + "="*70)
    print("TRAINING ORCHESTRATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
