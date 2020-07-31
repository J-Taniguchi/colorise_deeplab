import tensorflow as tf
import numpy as np
import h5py
# import numba


def make_xy_from_data_paths(x_paths,
                            y_paths,
                            image_size,
                            label,
                            extra_x_paths=None,
                            as_np=False):
    """make x and y from data paths.

    Args:
        x_paths (list): list of path to x image
        y_paths (list): list of path to y image or json. if None, y is exported as None
        image_size (tuple): model input and output size.(width, height)
        label (Label): class "Label" written in label.py
        data_type (str): select "image" or "index_png" or "polygon"

    Returns:
        np.rray, np.array: x (,extra_x, y)

    """
    x = []
    for i, x_path in enumerate(x_paths):
        image = tf.io.read_file(x_path)
        image = tf.image.decode_image(image, channels=3)
        out = np.zeros((image_size[1], image_size[0], 3))
        out[0:image.shape[0], 0:image.shape[1]] = image[:, :]
        x.append(out)
    if as_np:
        x = np.array(x, dtype=np.float32)
    else:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    if y_paths is None:
        return x

    y = []
    for i, y_path in enumerate(y_paths):
        if y_path is None:
            y.append(np.zeros((*image_size[::-1], label.n_labels), np.int32))
            continue
        image = tf.io.read_file(y_path)
        image = tf.image.decode_image(image, channels=3)
        y0 = convert_image_array_to_y(image, label)
        y.append(y0)

    if as_np:
        y = np.array(y, dtype=np.float32)
    else:
        y = tf.convert_to_tensor(y, dtype=tf.float32)
    if extra_x_paths is None:
        return x, y
    else:
        extra_x = []
        for i, extra_x_path in enumerate(extra_x_paths):
            if type(extra_x_path) is str:
                out = np.load(extra_x_path)
            else:
                out = np.load(extra_x_path.numpy())
            if len(out.shape) == 2:
                out = out[:, :, np.newaxis]
            extra_x.append(out)
        if as_np:
            extra_x = np.array(extra_x, dtype=np.float32)
        else:
            extra_x = tf.convert_to_tensor(extra_x, dtype=tf.float32)

        return x, extra_x, y


def make_xy_from_data_path(x_path,
                           y_path,
                           image_size):
    """make x and y from data path.

    Args:
        x_path: path to x image
        y_path: y image. if None, y is exported as None
        image_size (tuple): model input and output size.(width, height)
        label (Label): class "Label" written in label.py
    Returns:
        np.array: x, extra_x, y

    """

    # make x
    image = tf.io.read_file(x_path)
    image = tf.image.decode_image(image, channels=3)
    x = np.zeros((image_size[1], image_size[0], 3))
    x[0:image.shape[0], 0:image.shape[1]] = image[:, :]
    x = tf.convert_to_tensor(x)

    # make y
    if y_path is None:
        y = None
    else:
        image = tf.io.read_file(y_path)
        image = tf.image.decode_image(image, channels=3)
        y = np.zeros((image_size[1], image_size[0], 3))
        y[0:image.shape[0], 0:image.shape[1]] = image[:, :]
        y = tf.convert_to_tensor(y)

    return x, y


def random_crop(image, out_size):
    image_size = image.size
    xmin = np.inf
    ymin = np.inf
    # if fig size is not enough learge, xmin or ymin set to 0.
    if out_size[0] >= image_size[0]:
        xmin = 0
    if out_size[1] >= image_size[1]:
        ymin = 0

    if xmin == np.inf:
        x_res = image_size[0] - out_size[0]
        xmin = np.random.choice(np.arange(0, x_res + 1))
    if ymin == np.inf:
        y_res = image_size[1] - out_size[1]
        ymin = np.random.choice(np.arange(0, y_res + 1))
    xmax = xmin + out_size[0]
    ymax = ymin + out_size[1]
    return image.crop((xmin, ymin, xmax, ymax))


def get_random_crop_area(image_size, out_size):
    xmin = np.inf
    ymin = np.inf
    # if fig size is not enough learge, xmin or ymin set to 0.
    if out_size[0] >= image_size[0]:
        xmin = 0
    if out_size[1] >= image_size[1]:
        ymin = 0

    if xmin == np.inf:
        x_res = image_size[0] - out_size[0]
        xmin = np.random.choice(np.arange(0, x_res + 1))
    if ymin == np.inf:
        y_res = image_size[1] - out_size[1]
        ymin = np.random.choice(np.arange(0, y_res + 1))
    xmax = xmin + out_size[0]
    ymax = ymin + out_size[1]
    return (xmin, ymin, xmax, ymax)


def save_inference_results(fpath, pred, last_activation):
    with h5py.File(fpath, "w") as f:
        f.create_dataset("pred", data=pred)
        f.create_dataset("last_activation", data=last_activation)


def load_inference_results(fpath):
    with h5py.File(fpath, "r") as f:
        pred = f["pred"][()]
        last_activation = f["last_activation"][()]

    return pred, last_activation
