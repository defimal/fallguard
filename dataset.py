import tensorflow as tf
import pandas as pd
import os

IMG_SIZE = 224
CLASS_MAP = {"fall": 0, "stand": 1}

def load_dataset(csv_path, images_dir, batch_size=16, shuffle=True):
    df = pd.read_csv(csv_path)

    image_paths = df["filename"].apply(lambda x: os.path.join(images_dir, x)).values
    boxes = df[["xmin", "ymin", "xmax", "ymax"]].values
    sizes = df[["width", "height"]].values
    labels = df["class"].map(CLASS_MAP).values

    def _load_sample(path, box, size, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        box = tf.cast(box, tf.float32)
        w, h = tf.cast(size[0], tf.float32), tf.cast(size[1], tf.float32)

        box = box / tf.stack([w, h, w, h])

        return img, {"bbox": box, "class": label}

    ds = tf.data.Dataset.from_tensor_slices((image_paths, boxes, sizes, labels))
    ds = ds.map(_load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
