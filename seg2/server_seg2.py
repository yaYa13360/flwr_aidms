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

from typing import Dict, Optional, Tuple
from pathlib import Path

import flwr as fl
import tensorflow as tf

import timeit
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix


def main() -> None:
    # (segmentation2-----------------------------------------------------------
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

    
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.8,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )
    
    start_time = timeit.default_timer()
    # Start Flower server (SSL-enabled) for four rounds of federated learning
    accuracy_history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
        certificates=(
            Path("certificates/cache/ca.crt").read_bytes(),
            Path("certificates/cache/server.pem").read_bytes(),
            Path("certificates/cache/server.key").read_bytes(),
        ),
    )    
    end_time = timeit.default_timer() 
    elapsed = end_time - start_time
    print('########## elapsed ##########')    
    print(elapsed)

    # plot-----------------------------------------------------------
    data1 = accuracy_history.losses_centralized
    _, y_values = zip(*data1)
    plt.plot(y_values, linestyle='-', label = 'loss')
    plt.legend()
    plt.title('loss curve')
    plt.savefig("result/loss.png")
    plt.clf()
    
    data = accuracy_history.metrics_centralized['accuracy']
    _, y_values = zip(*data)
    plt.plot(y_values, linestyle='-', label = 'acc')
    plt.legend()
    plt.title('accuracy curve')
    plt.savefig("result/accuracy.png")
    plt.clf()

    data2 = accuracy_history.metrics_centralized['dice_coefficient']
    _, y_values = zip(*data2)
    plt.plot(y_values, linestyle='-', label = 'dice')
    plt.legend()
    plt.title('dice curve')
    plt.savefig("result/dice.png")
    plt.clf()
 
def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # (segmentation2)-----------------------------------------------------------
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
    
    val_ds, metadata = tfds.load(
        'oxford_iiit_pet',
        split='test',
        with_info=True,
    )
    
    val_images = val_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    BATCH_SIZE = 64
    
    validation_dataset = val_images.batch(BATCH_SIZE)
        

    # The `evaluate` function will be called after every round
    # only acc, loss ----------------------------------------------------------
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        
        loss, accuracy = model.evaluate(validation_dataset)

        # Calculate Dice coefficient
        dice_coefficient = calculate_dice_coefficient(model, validation_dataset)
        return loss, {"accuracy": accuracy, "dice_coefficient": dice_coefficient}

    return evaluate

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask

def dice_coef(y_true, y_pred, smooth=100):
    y_true_np = y_true.numpy()
    y_pred_np = create_mask(y_pred).numpy()
    dice = 0
    for task_id in list(np.unique(y_true)):
        y_true_task = (y_true_np == task_id).astype(np.float32)
        y_pred_task = (y_pred_np == task_id).astype(np.float32)
        intersection = np.sum(y_true_task * y_pred_task)
        union = np.sum(y_true_task) + np.sum(y_pred_task)
        dice += (2. * intersection + smooth) / (union + smooth)
    dice = dice/len(np.unique(y_true))
    return dice

def calculate_dice_coefficient(model, validation_dataset):
    dice_coefficient = 0.0
    num_samples = 0
    
    for batch in validation_dataset:
        images, masks = batch
        predictions = model.predict(images)
        for i in range(len(images)):
            dice_coefficient += dice_coef(masks[i], predictions[i])
            num_samples += 1
    
    return dice_coefficient / num_samples

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "local_epochs": 10 if server_round < 5 else 20,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 10 if server_round < 5 else 20
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
