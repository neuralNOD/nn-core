# -*- encoding: utf-8 -*-

"""
The Core Functionality to Train a Neural Network Model

A step-by-step generic function is developed here, which considers a
neural network model (specifically `tensorflow.keras` model, however,
it may also work with other model) trains the network, and save the
model file. The class function also provides functions for prediction
and trainings.
"""


class base(object):

    def __init__(self, model) -> None:
        self.model = model


    def compile(self, optimizer, loss, metrics) -> None:
        self.model.compile(
            optimizer = optimizer,
            loss = loss, metrics = metrics
        )


    def train(self, x, y, num_epochs, **kwargs) -> dict:
        shuffle = kwargs.get("shuffle", True)
        batch_size = kwargs.get("batch_size", 1024)
        validation_split = kwargs.get("validation_split", 0.05)

        history = self.model.fit(
            x, y, epochs = num_epochs,
            shuffle = shuffle, batch_size = batch_size,
            validation_split = validation_split
        )

        return history
