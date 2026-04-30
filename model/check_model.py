import tensorflow as tf
from pathlib import Path
p = Path(__file__).resolve().parent / 'waste_classifier_model.keras'
print('MODEL_EXISTS', p.exists())
if p.exists():
    model = tf.keras.models.load_model(str(p))
    print('OUTPUT_SHAPE', model.output_shape)
    print('LAST_LAYER', type(model.layers[-1]).__name__, model.layers[-1].name)
    act = getattr(model.layers[-1], 'activation', None)
    print('ACTIVATION', act.__name__ if act else None)
    print('NUM_LAYERS', len(model.layers))
