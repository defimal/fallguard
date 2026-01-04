import tensorflow as tf
from dataset import load_dataset
from model import build_model

def iou_metric(y_true, y_pred):
    # y_true, y_pred shape: (batch, 4) in [0,1]
    xA = tf.maximum(y_true[:, 0], y_pred[:, 0])
    yA = tf.maximum(y_true[:, 1], y_pred[:, 1])
    xB = tf.minimum(y_true[:, 2], y_pred[:, 2])
    yB = tf.minimum(y_true[:, 3], y_pred[:, 3])

    inter = tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA)

    area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])

    union = area_true + area_pred - inter + 1e-6
    return tf.reduce_mean(inter / union)


# Load datasets
train_ds = load_dataset(
    "data/annotations/train_annotations.csv",
    "data/images/train",
    batch_size=16
)

val_ds = load_dataset(
    "data/annotations/test_annotations.csv",
    "data/images/test",
    batch_size=16,
    shuffle=False
)

# Build model
model = build_model()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        "bbox": tf.keras.losses.Huber(),
        "class": tf.keras.losses.SparseCategoricalCrossentropy()
    },
    metrics={
        "class": ["accuracy"],
        "bbox": [iou_metric]
    }
)


model.summary()

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Save model
model.save("fallguard_model")
