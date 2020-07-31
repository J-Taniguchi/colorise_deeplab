# import numpy as np
import tensorflow as tf
# import tensorflow.keras as keras
from .data_utils import make_xy_from_data_path
from .data_augment import data_augment


def make_generator(img_paths,
                   image_size,
                   seg_img_paths=None,
                   preprocess=None,
                   augmentation=True,
                   ):

    def map_f(x_path, y_path=None):
        x, y = make_xy_from_data_path(
            x_path,
            y_path,
            image_size)

        if augmentation is True:
            # image_size for data_augment is (height, width)
            x, y = data_augment(
                x,
                y,
                image_size=image_size,
                p=0.95)

        x = tf.cast(x, tf.float32)

        if y is not None:
            y = tf.cast(y, tf.float32)
            if preprocess is None:
                y = (y / 127.5) - 1
            else:
                y = preprocess(y)

        if preprocess is None:
            x = (x / 127.5) - 1
        else:
            x = preprocess(x)

        if y_path is None:
            return x
        else:
            return x, y

    def wrap_mapf(x_path, y_path=None):
        if y_path is None:
            x_out = tf.py_function(
                map_f,
                inp=[x_path],
                Tout=(tf.float32))
            return x_out
        else:
            x_out, y_out = tf.py_function(
                map_f,
                inp=[x_path, y_path],
                Tout=(tf.float32, tf.float32))
            return x_out, y_out

    if seg_img_paths is None:
        ds = tf.data.Dataset.from_tensor_slices((img_paths))
    else:
        ds = tf.data.Dataset.from_tensor_slices((img_paths, seg_img_paths))

    return ds, wrap_mapf
