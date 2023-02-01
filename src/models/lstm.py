# -*- encoding: utf-8 -*-

"""
Long Short Term Memory (LSTM) Network Definations

A comprehensive functionality to build an effective and an
efficient LSTM network for training and validations.
"""

from typing import Iterable
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
)

class BareLSTM(object):
    def __init__(
        self,
        input_shape : Iterable[int],
        output_shape : int,
        **kwargs
    ) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape

        # get and define additional keyword arguments
        self.activation = kwargs.get("activation", None)
        self.model_name = kwargs.get("model_name", "BareLSTM-1.0.0")


    def get_2_layer_lstm(
        self,
        units : Iterable[int] = (64, 32),
        dropout : float = 0.2
    ):
        if len(units) != 2:
            raise ValueError("This is func. is for two layer LSTM network.")

        model = Sequential([
            LSTM(
                units[0],
                activation = self.activation,
                input_shape = self.input_shape,
                return_sequences = True,
                name = "iLayer"
            ),

            # the network consits of only one single hidden layer
            LSTM(units[1], activation = self.activation, recurrent_dropout = dropout, name = "HL-1"),
            # Dense(int(self.output_shape * 2), name = "HL-2"),
            # Dropout(dropout, name = "HL-2-D"),

            Dense(self.output_shape, name = "oLayer")
        ], name = self.model_name)
        return model
