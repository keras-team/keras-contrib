import warnings

import h5py
import keras.backend as K
from keras import optimizers
from keras.engine import saving


def save_all_weights(model, filepath, include_optimizer=True):
    """
    Save model weights and optimizer weights but not configuration to a HDF5 file.
    Functionally between `save` and `save_weights`.

    The HDF5 file contains:
        - the model's weights
        - the model's optimizer's state (if any)
    If you have a complicated model or set of models that do not serialize
    to JSON correctly, use this method.
    # Arguments
        model: Keras model instance to be saved.
        filepath: String, path where to save the model.
        include_optimizer: If True, save optimizer's state together.
    # Raises
        ImportError: if h5py is not available.
    """
    if h5py is None:
        raise ImportError('`save_all_weights` requires h5py.')

    with h5py.File(filepath, 'w') as f:
        model_weights_group = f.create_group('model_weights')
        model_layers = model.layers
        saving.save_weights_to_hdf5_group(model_weights_group, model_layers)

        if include_optimizer and hasattr(model, 'optimizer') and model.optimizer:
            if isinstance(model.optimizer, optimizers.TFOptimizer):
                warnings.warn(
                    'TensorFlow optimizers do not '
                    'make it possible to access '
                    'optimizer attributes or optimizer state '
                    'after instantiation. '
                    'As a result, we cannot save the optimizer '
                    'as part of the model save file.'
                    'You will have to compile your model again after loading it. '
                    'Prefer using a Keras optimizer instead '
                    '(see keras.io/optimizers).')
            else:
                # Save optimizer weights.
                symbolic_weights = getattr(model.optimizer, 'weights')
                if symbolic_weights:
                    optimizer_weights_group = f.create_group('optimizer_weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    weight_names = []
                    for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                        # Default values of symbolic_weights is /variable for theano
                        if K.backend() == 'theano':
                            if hasattr(w, 'name') and w.name != "/variable":
                                name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        else:
                            if hasattr(w, 'name') and w.name:
                                name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        weight_names.append(name.encode('utf8'))
                    optimizer_weights_group.attrs['weight_names'] = weight_names
                    for name, val in zip(weight_names, weight_values):
                        param_dset = optimizer_weights_group.create_dataset(
                            name,
                            val.shape,
                            dtype=val.dtype)
                        if not val.shape:
                            # scalar
                            param_dset[()] = val
                        else:
                            param_dset[:] = val


def load_all_weights(model, filepath, include_optimizer=True):
    """Loads the weights of a model saved via `save_all_weights`.
    If model has been compiled, optionally load its optimizer's weights.
    # Arguments
        model: instantiated model with architecture matching the saved model.
            Compile the model beforehand if you want to load optimizer weights.
        filepath: String, path to the saved model.
    # Returns
        None. The model will have its weights updated.
    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError('`load_all_weights` requires h5py.')

    with h5py.File(filepath, mode='r') as f:
        # set weights
        saving.load_weights_from_hdf5_group(f['model_weights'], model.layers)
        # Set optimizer weights.
        if (include_optimizer
                and 'optimizer_weights' in f and hasattr(model, 'optimizer')
                and model.optimizer):
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [n.decode('utf8') for n in
                                      optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]
            model.optimizer.set_weights(optimizer_weight_values)
