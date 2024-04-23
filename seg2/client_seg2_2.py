# import argparse
import os
import cv2
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import imantics
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import os
from pathlib import Path

import tensorflow as tf

import flwr as fl

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, train): # (segmentation)
        self.model = model
        self.train = train

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
            
        # Train the model using hyperparameters from config (segmentation)
        history = self.model.fit(
            self.train
#            epochs,
#            verbose = 1
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
#            "val_loss": history.history["val_loss"][0],
#            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

#     def evaluate(self, parameters, config):
#         """Evaluate parameters on the locally held test set."""

#         # Update local model with global parameters
#         self.model.set_weights(parameters)

#         # Get config values
#         steps: int = config["val_steps"]

#         # Evaluate global model parameters on the local test data and return results
#         loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
#         num_examples_test = len(self.x_test)
#         return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # (segmentation2) model-----------------------------------------------------------
    def normalize(input_image, input_mask):
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    def load_image(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(
            datapoint['segmentation_mask'],
            (128, 128),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask
    
    (train_ds1, train_ds2), metadata = tfds.load(
        'oxford_iiit_pet',
        split=['train[:50%]', 'train[50%:]'],
        with_info=True,
        )
    
    train_images = train_ds1.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    BATCH_SIZE = 64
    
    train_dataset = (
        train_images
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    def unet_model(output_channels:int):
      inputs = tf.keras.layers.Input(shape=[128, 128, 3])

      # Downsampling through the model
      skips = down_stack(inputs)
      x = skips[-1]
      skips = reversed(skips[:-1])

      # Upsampling and establishing the skip connections
      for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

      # This is the last layer of the model
      last = tf.keras.layers.Conv2DTranspose(
          filters=output_channels, kernel_size=3, strides=2,
          padding='same')  #64x64 -> 128x128

      x = last(x)

      return tf.keras.Model(inputs=inputs, outputs=x)

    OUTPUT_CLASSES = 3

    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    
    ##########################################################

    # Start Flower client
    client = CifarClient(model, train_dataset)

    fl.client.start_numpy_client(
        server_address="#",
        client=client,
        root_certificates=Path("certificates/cache/ca.crt").read_bytes(),
    )


if __name__ == "__main__":
    main()
