"""Example application with a dummy model using callback-based Neptune integration.
"""
import numpy as np

from keras_contrib import callbacks
from keras.models import Sequential
from keras.layers import Dense


# Replace below with your own credentials
# If NEPTUNE_API_TOKEN is None, read from environment variable
#  otherwise you should provide it as a str
PROJECT_QUALIFIED_NAME = "your_username/your_project"
EXPERIMENT_NAME = "test-neptunelogger"
NEPTUNE_API_TOKEN = None


def build_model():
    """Build a dummy binary classification model model.

    Returns:
        Keras.models.Model: The dummy model.
    """
    model = Sequential([
        Dense(2, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    return model


def test_NeptuneLogger():
    """Test the NeptuneLogger callback with a dummy model and dataset.
    """
    X = np.random.rand(100, 2)
    y = np.random.rand(100).reshape(-1, 1)

    model = build_model()
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        X, y,
        batch_size=1,
        epochs=1,
        verbose=0,
        callbacks=[
            callbacks.NeptuneLogger(
                project_qualified_name=PROJECT_QUALIFIED_NAME,
                experiment_name=EXPERIMENT_NAME,
                api_token=NEPTUNE_API_TOKEN
            )
        ])

if __name__ == '__main__':
    test_NeptuneLogger()
