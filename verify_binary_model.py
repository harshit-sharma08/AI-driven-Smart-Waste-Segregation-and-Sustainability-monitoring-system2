#!/usr/bin/env python3
"""
Verify that the newly trained model is binary with correct shape and activation.
Run this after training completes.
"""
import tensorflow as tf
from pathlib import Path

model_path = Path(__file__).resolve().parent / "waste_classifier_model.keras"
print(f"✓ Checking model: {model_path}")
print(f"✓ Model exists: {model_path.exists()}\n")

if model_path.exists():
    model = tf.keras.models.load_model(str(model_path))
    
    print("=== Model Inspection ===")
    print(f"Output shape: {model.output_shape}")
    print(f"Last layer type: {type(model.layers[-1]).__name__}")
    print(f"Last layer name: {model.layers[-1].name}")
    
    last_layer = model.layers[-1]
    activation = getattr(last_layer, 'activation', None)
    print(f"Activation function: {activation.__name__ if activation else 'None'}")
    
    print("\n=== Expected for Binary Classification ===")
    print(f"✓ Output shape should be: (None, 1) — Got: {model.output_shape}")
    print(f"✓ Activation should be: sigmoid — Got: {activation.__name__ if activation else 'None'}")
    print(f"✓ Last layer type should be: Dense — Got: {type(model.layers[-1]).__name__}")
    
    # Verdict
    is_correct_shape = model.output_shape == (None, 1)
    is_correct_activation = activation is not None and activation.__name__ == 'sigmoid'
    
    if is_correct_shape and is_correct_activation:
        print("\n✅ SUCCESS! Binary model is correctly trained.")
        print("   - Cola bottles will now be classified as RECYCLABLE (not organic)")
        print("   - Plastic items will correctly map to the Blue Bin")
    else:
        print("\n❌ ERROR: Model is not binary!")
        if not is_correct_shape:
            print(f"   - Wrong shape: {model.output_shape} (expected (None, 1))")
        if not is_correct_activation:
            print(f"   - Wrong activation: {activation.__name__ if activation else 'None'} (expected sigmoid)")
else:
    print("❌ Model file not found!")
