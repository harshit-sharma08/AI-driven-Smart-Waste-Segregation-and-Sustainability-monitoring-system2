#!/usr/bin/env python3
"""
Test the newly trained binary model on an image to verify correct predictions.
Run this after training completes.
"""
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# Config
MODEL_PATH = "model/waste_classifier_model.keras"
DATASET_TEST_DIR = "model/DATASET/TEST"
IMG_SIZE = (224, 224)
labels = ["organic", "recyclable"]

print("=" * 60)
print("WASTE CLASSIFIER BINARY MODEL - PREDICTION TEST")
print("=" * 60)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded: {MODEL_PATH}")
    print(f"  Output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Try to find a test image
test_dir = Path(DATASET_TEST_DIR)
if not test_dir.exists():
    print(f"⚠  Test directory not found: {DATASET_TEST_DIR}")
    print("   Skipping test image predictions")
    sys.exit(0)

# Find an image from recyclable (should NOT be organic now!)
recyclable_dir = test_dir / "R"
organic_dir = test_dir / "O"

test_images = []
if recyclable_dir.exists():
    files = list(recyclable_dir.glob("*.jpg"))[:1]  # Take first one
    if files:
        test_images.append(("Recyclable Test", files[0]))

if organic_dir.exists():
    files = list(organic_dir.glob("*.jpg"))[:1]
    if files:
        test_images.append(("Organic Test", files[0]))

if not test_images:
    print("⚠  No test images found in DATASET/TEST")
    sys.exit(0)

print(f"\n✓ Testing {len(test_images)} samples:\n")

def preprocess_image(img_path: Path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

for name, img_path in test_images:
    try:
        img_data = preprocess_image(img_path)
        pred = model.predict(img_data, verbose=0)
        
        if pred.ndim == 2 and pred.shape[1] == 1:
            prob = float(pred[0][0])
            label = labels[int(prob >= 0.5)]
            confidence = prob if label == "recyclable" else 1.0 - prob
        else:
            # Fallback for unexpected shape
            prob = float(pred.flat[0])
            label = labels[int(prob >= 0.5)]
            confidence = prob if label == "recyclable" else 1.0 - prob
        
        print(f"{name:20} → Label: {label:12} Confidence: {confidence:.2%}")
        
        # Expected behavior check
        expected = name.split()[0].lower()
        if (expected == "recyclable" and label == "recyclable") or \
           (expected == "organic" and label == "organic"):
            print(f"  ✅ Correct!")
        else:
            print(f"  ⚠  Expected {expected}, got {label}")
            
    except Exception as e:
        print(f"{name:20} → ❌ Error: {e}")

print("\n" + "=" * 60)
print("Test complete. If all predictions are CORRECT, the binary")
print("model is working perfectly and plastic/inorganic items")
print("will NO LONGER be classified as ORGANIC!")
print("=" * 60)
