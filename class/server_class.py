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
    # Load and compile Keras model (classification)-----------------------------------------------------------
    
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    

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
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
        certificates=(
            Path("certificates/cache/ca.crt").read_bytes(),
            Path("certificates/cache/server.pem").read_bytes(),
            Path("certificates/cache/server.key").read_bytes(),
        ),
    )    
    end_time = timeit.default_timer() 
    elapsed = end_time - start_time

#     print(accuracy_history.losses_centralized) 
#     print(accuracy_history.metrics_centralized['accuracy']) 
    
    # plot-----------------------------------------------------------
    data1 = accuracy_history.losses_centralized
    _, y_values = zip(*data1)
    plt.plot(y_values, linestyle='-', label = 'loss')
    plt.legend()
    plt.title('loss curve')
    plt.savefig("loss.png")
    plt.clf()
    
    data = accuracy_history.metrics_centralized['accuracy']
    _, y_values = zip(*data)
    plt.plot(y_values, linestyle='-', label = 'acc')
    plt.legend()
    plt.title('accuracy curve')
    plt.savefig("certificates/cache/accuracy.png")
    plt.clf()
    
    
    print('########## elapsed ##########')    
    print(elapsed)

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself (classification) -------------------------
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # Use the last 5k training examples as a validation set
    x_val, y_val = x_train[45000:50000], y_train[45000:50000]
    
    # The `evaluate` function will be called after every round
    # only acc, loss ----------------------------------------------------------
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        
        loss, accuracy = model.evaluate(x_val, y_val)
        
        return loss, {"accuracy": accuracy}

    return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 4,
        "local_epochs": 5 if server_round < 20 else 10,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 20 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
