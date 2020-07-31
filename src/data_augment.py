import tensorflow as tf
# import numpy as np


def augmentor(in_image, out_image):
    # you can edit here for image augmentation.
    # image and mask are 3 dim tensors.
    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + 0.5), tf.bool)
    in_image, out_image = tf.cond(
        should_apply_op,
        lambda: (tf.image.flip_left_right(in_image),
                 tf.image.flip_left_right(out_image)),
        lambda: (in_image, out_image))

    should_apply_op = tf.cast(
        tf.floor(tf.random.uniform([], dtype=tf.float32) + 0.5), tf.bool)
    in_image, out_image = tf.cond(
        should_apply_op,
        lambda: (tf.image.flip_up_down(in_image),
                 tf.image.flip_up_down(out_image)),
        lambda: (in_image, out_image))

    # out_image = tf.image.random_hue(out_image, 0.05)
    # out_image = tf.image.random_saturation(out_image, 0.9, 1.1)
    # out_image = tf.image.random_brightness(out_image, 0.1)
    # in_image = tf.image.random_brightness(in_image, 0.1)
    # image = tf.image.random_jpeg_quality(image, 90, 100)
    return in_image, out_image


def data_augment(in_image, out_image, image_size, p):
    # if p < 0.0:
    #    raise ValueError("p < 0. p must be positive number.")
    # if len(image.shape) != 3:
    #    raise Exception("dimension of images for data_augment must be 3")
    # if len(mask.shape) != 3:
    #    raise Exception("dimension of masks for data_augment must be 3")
    p = float(p)
    should_apply_op = tf.cast(tf.floor(tf.random.uniform([], dtype=tf.float32) + p), tf.bool)

    in_image, out_image = tf.cond(
        should_apply_op,
        lambda: augmentor(in_image, out_image),
        lambda: (in_image, out_image))

    return in_image, out_image
