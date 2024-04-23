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

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    
#     def __init__(self, model, x_train, y_train, x_test, y_test):
    def __init__(self, model, x_train, y_train): # (classification)
        self.model = model
        self.x_train, self.y_train = x_train, y_train
#         self.x_test, self.y_test = x_test, y_test
        
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
    
        # Train the model using hyperparameters from config (classification)
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.3,
        )

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
   
    # Load and compile Keras model (classification)-----------------------------------------------------------
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load a subset of CIFAR-10 to simulate the local data partition
    ##########################################################
    (x_total, y_total), _ = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = x_total[22500:45000], y_total[22500:45000]

    
    ##########################################################

    # Start Flower client
    client = CifarClient(model, x_train, y_train)

    fl.client.start_numpy_client(
        server_address="#",
        client=client,
        root_certificates=Path("certificates/cache/ca.crt").read_bytes(),
    )


if __name__ == "__main__":
    main()
