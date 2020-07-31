import os
import sys
import yaml
import shutil

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)
use_devices = str(conf["use_devices"])
os.environ["CUDA_VISIBLE_DEVICES"] = use_devices
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from src.model import deeplab_v3plus_transfer_os16
from src.input_data_processing import make_dataset
import src.mod_xception as xception
tf.compat.v1.enable_eager_execution()
matplotlib.use('Agg')

out_dir = conf["model_dir"]

train_x_dirs = conf["train_x_dirs"]
train_y_dirs = conf["train_y_dirs"]

valid_x_dirs = conf["valid_x_dirs"]
valid_y_dirs = conf["valid_y_dirs"]

batch_size = conf["batch_size"]
n_epochs = conf["n_epochs"]
image_size = conf["image_size"]
optimizer = conf["optimizer"]

use_tensorboard = conf.get("use_tensorboard", False)
use_batch_renorm = conf.get("use_batch_renorm", False)

n_gpus = len(use_devices.split(','))

batch_size = batch_size * n_gpus

os.makedirs(out_dir, exist_ok=True)

preprocess = keras.applications.xception.preprocess_input

# make train dataset
train_dataset, train_path_list = make_dataset(
    train_x_dirs,
    image_size,
    preprocess,
    batch_size,
    y_dirs=train_y_dirs,
    data_augment=True,
    shuffle=True
)

# make valid dataset
valid_dataset, valid_path_list = make_dataset(
    valid_x_dirs,
    image_size,
    preprocess,
    batch_size,
    y_dirs=valid_y_dirs,
    data_augment=False,
    shuffle=False
)

# define loss function
loss_function = keras.losses.MSE

# define optimizer
if optimizer == "Adam":
    opt = tf.keras.optimizers.Adam()
elif optimizer == "Nadam":
    opt = tf.keras.optimizers.Nadam()
elif optimizer == "SGD":
    opt = tf.keras.optimizers.SGD()
else:
    raise Exception(
        "optimizer " + optimizer + " is not supported")

# make model
layer_name_to_decoder = "block3_sepconv2_bn"
encoder_end_layer_name = "block13_sepconv2_bn"

if n_gpus >= 2:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        encoder = xception.Xception(
            input_shape=(*image_size, 3),
            weights="imagenet",
            include_top=False)

        model = deeplab_v3plus_transfer_os16(
            encoder,
            layer_name_to_decoder,
            encoder_end_layer_name,
            freeze_encoder=False,
            batch_renorm=use_batch_renorm)

        model.compile(optimizer=opt,
                      loss=loss_function,
                      run_eagerly=True,
                      )
else:
    encoder = xception.Xception(
        input_shape=(*image_size, 3),
        weights="imagenet",
        include_top=False)
    model = deeplab_v3plus_transfer_os16(
        encoder,
        layer_name_to_decoder,
        encoder_end_layer_name,
        freeze_encoder=False,
        batch_renorm=use_batch_renorm,
    )

    model.compile(optimizer=opt,
                  loss=loss_function,
                  run_eagerly=True,
                  )
model.summary()

filepath = os.path.join(out_dir, 'best_model.h5')
cp_cb = keras.callbacks.ModelCheckpoint(
    filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min')
cbs = [cp_cb]

if use_tensorboard:
    log_dir = os.path.join(out_dir, "logs")
    TB_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, write_graph=False)
    cbs.append(TB_cb)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

# training
n_train_data = len(train_path_list["x"])
n_valid_data = len(valid_path_list["x"])

n_train_batch = int(np.ceil(n_train_data / batch_size))
n_valid_batch = int(np.ceil(n_valid_data / batch_size))
print("train batch:{}".format(n_train_batch))
print("valid batch:{}".format(n_valid_batch))
hist = model.fit(
    train_dataset,
    epochs=n_epochs,
    validation_data=valid_dataset,
    callbacks=cbs,
)

# write log
hists = hist.history
hists_df = pd.DataFrame(hists)
hists_df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(hists_df["loss"], label="loss")
plt.plot(hists_df["val_loss"], label="val_loss")
plt.yscale("log")
plt.legend()
plt.grid(b=True)

plt.savefig(os.path.join(out_dir, 'losscurve.png'))

model.save(os.path.join(out_dir, 'final_epoch.h5'))
