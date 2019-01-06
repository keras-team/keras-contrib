"""Functional tools for working with layers."""

from functools import reduce

__all__ = ['sequence', 'repeat']


def sequence(*layers):
    """Composes layers sequentially.

    # Arguments
        *layers: Layers, or other callables that map a tensor to a tensor.

    # Returns
        A callable that maps a tensor to the output tensor of the last layer.

    # Examples

    ```python
        from keras.layers import Dense, Input
        from keras.models import Model
        from keras_contrib.layers import sequence

        input_layer = Input(shape=(16,))

        output = sequence(
            Dense(8, activation='relu'),
            Dense(8, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1),
        )(input_layer)

        model = Model(input_layer, output)
    ```
    """
    return reduce(lambda f, g: lambda x: g(f(x)), layers, lambda x: x)


def repeat(n, layer_factory):
    """Constructs a sequence of repeated layers.

    # Arguments
        n: int. The number of times to repeat the layer.
        layer_factory: A function taking no arguments that returns a layer or
            another callable that maps a tensor to a tensor.

    # Returns
        A callable that maps a tensor to the output tensor of the last layer.

    # Examples

    ```python
        from keras.layers import Dense, Input
        from keras.models import Model
        from keras_contrib.layers import repeat, sequence

        input_layer = Input(shape=(16,))

        output = sequence(
            repeat(3, lambda: Dense(8, activation='relu')),
            Dense(1),
        )(input_layer)

        model = Model(input_layer, output)
    ```

    `sequence` and `repeat` can be freely intermixed with layers, since they
    both map a tensor to a tensor:

    ```python
        from keras.layers import Activation, Dense, Input
        from keras.models import Model
        from keras_contrib.layers import repeat, sequence

        input_layer = Input(shape=(16,))

        output = sequence(
            repeat(3, lambda: sequence(
                Dense(8),
                Activation('relu'),
            )),
            Dense(1),
        )(input_layer)

        model = Model(input_layer, output)
    ```
    """
    return sequence(*(layer_factory() for _ in range(n)))
