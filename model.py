import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 2

def build_model():
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    for layer in backbone.layers[-40:]:
        layer.trainable = True


    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)

    # Bounding box head
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox")(x)

    # Classification head
    class_output = layers.Dense(NUM_CLASSES, activation="softmax", name="class")(x)

    model = models.Model(
        inputs=backbone.input,
        outputs={"bbox": bbox_output, "class": class_output}
    )

    return model
