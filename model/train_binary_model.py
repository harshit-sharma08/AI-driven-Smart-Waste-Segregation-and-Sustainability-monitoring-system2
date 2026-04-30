import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_ONEDNN'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, 'DATASET', 'TRAIN')
TEST_DIR = os.path.join(BASE_DIR, 'DATASET', 'TEST')
OUTPUT_MODEL = os.path.join(BASE_DIR, 'waste_classifier_model.keras')
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 123
AUTO = tf.data.AUTOTUNE

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
print('TF_ENABLE_ONEDNN_OPTS =', os.environ.get('TF_ENABLE_ONEDNN_OPTS'))
print('TF_ENABLE_ONEDNN =', os.environ.get('TF_ENABLE_ONEDNN'))
print('OMP_NUM_THREADS =', os.environ.get('OMP_NUM_THREADS'))
print('MKL_NUM_THREADS =', os.environ.get('MKL_NUM_THREADS'))
print('Intra-op threads =', tf.config.threading.get_intra_op_parallelism_threads())
print('Inter-op threads =', tf.config.threading.get_inter_op_parallelism_threads())
print('Batch size =', BATCH_SIZE)

print('TRAIN DIR:', TRAIN_DIR)
print('TEST DIR:', TEST_DIR)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='binary',
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

print('Classes:', train_ds.class_names)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='binary',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

train_ds = train_ds.cache().prefetch(buffer_size=AUTO)
val_ds = val_ds.cache().prefetch(buffer_size=AUTO)
test_ds = test_ds.cache().prefetch(buffer_size=AUTO)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = layers.RandomFlip('horizontal')(inputs)
x = layers.RandomRotation(0.15)(x)
x = layers.RandomZoom(0.15)(x)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]

print('Starting head training...')
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
)

print('Starting fine-tuning...')
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=callbacks,
)

print('Evaluating on test set...')
results = model.evaluate(test_ds, return_dict=True)
print('Test results:', results)

predictions = model.predict(test_ds)
probs = predictions.ravel()
labels = (probs >= 0.5).astype('int32')

true_labels = []
for _, y_batch in test_ds:
    true_labels.extend(y_batch.numpy().astype('int32').tolist())

assert len(true_labels) == len(labels)

count_correct = sum(int(p == t) for p, t in zip(labels, true_labels))
print('Test accuracy computed:', count_correct / len(labels))

counts = {
    'true_organic': int(sum(1 for t in true_labels if t == 0)),
    'true_recyclable': int(sum(1 for t in true_labels if t == 1)),
    'pred_organic': int(sum(1 for p in labels if p == 0)),
    'pred_recyclable': int(sum(1 for p in labels if p == 1)),
}
print('Counts:', counts)

confusion = {
    'organic_as_organic': int(sum(1 for p, t in zip(labels, true_labels) if p == 0 and t == 0)),
    'organic_as_recyclable': int(sum(1 for p, t in zip(labels, true_labels) if p == 1 and t == 0)),
    'recyclable_as_recyclable': int(sum(1 for p, t in zip(labels, true_labels) if p == 1 and t == 1)),
    'recyclable_as_organic': int(sum(1 for p, t in zip(labels, true_labels) if p == 0 and t == 1)),
}
print('Confusion:', confusion)

print('Saving model to', OUTPUT_MODEL)
model.save(OUTPUT_MODEL)
print('Saved model.')
