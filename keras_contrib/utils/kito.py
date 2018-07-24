"""
Reduce neural net structure (Conv + BN -> Conv)
Also works:
DepthwiseConv2D + BN -> DepthwiseConv2D
SeparableConv2D + BN -> SeparableConv2D

This code takes on input trained Keras model and optimize layer structure and weights in such a way
that model became much faster (~30%), but works identically to initial model. It can be extremely
useful in case you need to process large amount of images with trained model. Reduce operation was
tested on all Keras models zoo. See comparison table and full description by link:
https://github.com/ZFTurbo/Keras-inference-time-optimizer
Author: Roman Solovyev (ZFTurbo)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def get_keras_sub_version():
    from keras import __version__
    type = int(__version__.split('.')[1])
    return type


def get_input_layers_ids(model, layer, verbose):
    res = dict()
    for i, l in enumerate(model.layers):
        layer_id = str(id(l))
        res[layer_id] = i

    inbound_layers = []
    layer_id = str(id(layer))
    for i, node in enumerate(layer._inbound_nodes):
        node_key = layer.name + '_ib-' + str(i)
        if get_keras_sub_version() == 1:
            network_nodes = model._container_nodes
        else:
            network_nodes = model._network_nodes
        if node_key in network_nodes:
            for inbound_layer in node.inbound_layers:
                inbound_layer_id = str(id(inbound_layer))
                inbound_layers.append(res[inbound_layer_id])
    return inbound_layers


def get_output_layers_ids(model, layer, verbose):
    res = dict()
    for i, l in enumerate(model.layers):
        layer_id = str(id(l))
        res[layer_id] = i

    outbound_layers = []
    layer_id = str(id(layer))
    for i, node in enumerate(layer._outbound_nodes):
        node_key = layer.name + '_ib-' + str(i)
        if get_keras_sub_version() == 1:
            network_nodes = model._container_nodes
        else:
            network_nodes = model._network_nodes
        if node_key in network_nodes:
            outbound_layer_id = str(id(node.outbound_layer))
            if outbound_layer_id in res:
                outbound_layers.append(res[outbound_layer_id])
            else:
                print('Warning, some problem with outbound node on layer {}!'.format(layer.name))
    return outbound_layers


def get_copy_of_layer(layer, verbose):
    if get_keras_sub_version() == 1:
        from keras.applications.mobilenet import relu6
    else:
        from keras_applications.mobilenet import relu6
    from keras.layers.core import Activation
    from keras import layers
    config = layer.get_config()

    # Non-standard relu6 layer (from MobileNet)
    if layer.__class__.__name__ == 'Activation':
        if config['activation'] == 'relu6':
            layer_copy = Activation(relu6, name=layer.name)
            return layer_copy

    # DeepLabV3+ non-standard layer
    if layer.__class__.__name__ == 'BilinearUpsampling':
        from neural_nets.deeplab_v3_plus_model import BilinearUpsampling
        layer_copy = BilinearUpsampling(upsampling=config['upsampling'], output_size=config['output_size'], name=layer.name)
        return layer_copy

    layer_copy = layers.deserialize({'class_name': layer.__class__.__name__, 'config': config})
    layer_copy.name = layer.name
    return layer_copy


def get_layers_without_output(model, verbose):
    output_tensor = []
    output_names = []
    for level_id in range(len(model.layers)):
        output_layers = get_output_layers_ids(model, model.layers[level_id], verbose)
        if len(output_layers) == 0:
            output_tensor.append(model.layers[level_id].output)
            output_names.append(model.layers[level_id].name)
    if verbose:
        print('Outputs [{}]: {}'.format(len(output_tensor), output_names))
    return output_tensor, output_names


def copy_keras_model_low_level(model, verbose):
    from keras.models import Model

    x = None
    input = None
    tmp_model = None
    for level_id in range(len(model.layers)):
        layer = model.layers[level_id]
        layer_type = layer.__class__.__name__
        input_layers = get_input_layers_ids(model, layer, verbose)
        output_layers = get_output_layers_ids(model, layer, verbose)
        if verbose:
            print('Go for {}: {} ({}). Input layers: {} Output layers: {}'.format(level_id, layer_type, layer.name, input_layers, output_layers))
        if x is None:
            input = get_copy_of_layer(layer, verbose)
            x = input
            tmp_model = Model(inputs=input.output, outputs=x.output)
        else:
            new_layer = get_copy_of_layer(layer, verbose)

            prev_layer = []
            for i in range(len(input_layers)):
                tens = tmp_model.get_layer(name=model.layers[input_layers[i]].name).output
                prev_layer.append(tens)
            if len(prev_layer) == 1:
                prev_layer = prev_layer[0]

            output_tensor, output_names = get_layers_without_output(tmp_model, verbose)
            x = new_layer(prev_layer)
            if layer.name not in output_names:
                output_tensor.append(x)
            else:
                output_tensor = x
            tmp_model = Model(inputs=input.output, outputs=output_tensor)
            tmp_model.get_layer(name=layer.name).set_weights(layer.get_weights())

    model = Model(inputs=input.output, outputs=x)
    return model


def optimize_conv2d_batchnorm_block(m, initial_model, input_layers, conv, bn, verbose):
    from keras import layers
    from keras.models import Model

    conv_layer_type = conv.__class__.__name__
    conv_config = conv.get_config()
    conv_config['use_bias'] = True
    bn_config = bn.get_config()
    if conv_config['activation'] != 'linear':
        print('Only linear activation supported for conv + bn optimization!')
        exit()

    # Copy Conv2D layer
    layer_copy = layers.deserialize({'class_name': conv.__class__.__name__, 'config': conv_config})
    # We use batch norm name here to find it later
    layer_copy.name = bn.name

    # Create new model to initialize layer. We need to store other output tensors as well
    output_tensor, output_names = get_layers_without_output(m, verbose)
    input_layer_name = initial_model.layers[input_layers[0]].name
    prev_layer = m.get_layer(name=input_layer_name)
    x = layer_copy(prev_layer.output)

    output_tensor_to_use = [x]
    for i in range(len(output_names)):
        if output_names[i] != input_layer_name:
            output_tensor_to_use.append(output_tensor[i])

    if len(output_tensor_to_use) == 1:
        output_tensor_to_use = output_tensor_to_use[0]

    tmp_model = Model(inputs=m.input, outputs=output_tensor_to_use)

    if conv.get_config()['use_bias']:
        (conv_weights, conv_bias) = conv.get_weights()
    else:
        (conv_weights,) = conv.get_weights()

    if bn_config['scale']:
        gamma, beta, run_mean, run_std = bn.get_weights()
    else:
        gamma = 1.0
        beta, run_mean, run_std = bn.get_weights()

    eps = bn_config['epsilon']
    A = gamma / np.sqrt(run_std + eps)

    if conv.get_config()['use_bias']:
        B = beta + (gamma * (conv_bias - run_mean) / np.sqrt(run_std + eps))
    else:
        B = beta - ((gamma * run_mean) / np.sqrt(run_std + eps))

    if conv_layer_type == 'Conv2D':
        for i in range(conv_weights.shape[-1]):
            conv_weights[:, :, :, i] *= A[i]
    elif conv_layer_type == 'DepthwiseConv2D':
        for i in range(conv_weights.shape[-2]):
            conv_weights[:, :, i, :] *= A[i]

    tmp_model.get_layer(layer_copy.name).set_weights((conv_weights, B))
    return tmp_model


def optimize_separableconv2d_batchnorm_block(m, initial_model, input_layers, conv, bn, verbose=False):
    from keras import layers
    from keras.models import Model

    conv_config = conv.get_config()
    conv_config['use_bias'] = True
    bn_config = bn.get_config()
    if conv_config['activation'] != 'linear':
        print('Only linear activation supported for conv + bn optimization!')
        exit()

    layer_copy = layers.deserialize({'class_name': conv.__class__.__name__, 'config': conv_config})
    # We use batch norm name here to find it later
    layer_copy.name = bn.name

    # Create new model to initialize layer. We need to store other output tensors as well
    output_tensor, output_names = get_layers_without_output(m, verbose)
    input_layer_name = initial_model.layers[input_layers[0]].name
    prev_layer = m.get_layer(name=input_layer_name)
    x = layer_copy(prev_layer.output)

    output_tensor_to_use = [x]
    for i in range(len(output_names)):
        if output_names[i] != input_layer_name:
            output_tensor_to_use.append(output_tensor[i])

    if len(output_tensor_to_use) == 1:
        output_tensor_to_use = output_tensor_to_use[0]

    tmp_model = Model(inputs=m.input, outputs=output_tensor_to_use)

    if conv.get_config()['use_bias']:
        (conv_weights_3, conv_weights_1, conv_bias) = conv.get_weights()
    else:
        (conv_weights_3, conv_weights_1) = conv.get_weights()

    if bn_config['scale']:
        gamma, beta, run_mean, run_std = bn.get_weights()
    else:
        gamma = 1.0
        beta, run_mean, run_std = bn.get_weights()

    eps = bn_config['epsilon']
    A = gamma / np.sqrt(run_std + eps)

    if conv.get_config()['use_bias']:
        B = beta + (gamma * (conv_bias - run_mean) / np.sqrt(run_std + eps))
    else:
        B = beta - ((gamma * run_mean) / np.sqrt(run_std + eps))

    for i in range(conv_weights_1.shape[-1]):
        conv_weights_1[:, :, :, i] *= A[i]

    # print(conv_weights_3.shape, conv_weights_1.shape, A.shape)

    tmp_model.get_layer(layer_copy.name).set_weights((conv_weights_3, conv_weights_1, B))
    return tmp_model


def reduce_keras_model(model, verbose=False):
    from keras.models import Model

    x = None
    input = None
    tmp_model = None
    skip_layers = []
    keras_sub_version = get_keras_sub_version()
    if verbose:
        print('Keras sub version: {}'.format(keras_sub_version))

    for level_id in range(len(model.layers)):
        layer = model.layers[level_id]
        layer_type = layer.__class__.__name__
        input_layers = get_input_layers_ids(model, layer, verbose)
        output_layers = get_output_layers_ids(model, layer, verbose)
        if verbose:
            print('Go for {}: {} ({}). Input layers: {} Output layers: {}'.format(level_id, layer_type, layer.name, input_layers, output_layers))

        if level_id in skip_layers:
            if verbose:
                print('Skip layer because it was removed during optimization!')
            continue

        # Special cases for reducing
        if len(output_layers) == 1:
            next_layer = model.layers[output_layers[0]]
            next_layer_type = next_layer.__class__.__name__
            if layer_type in ['Conv2D', 'DepthwiseConv2D'] and next_layer_type == 'BatchNormalization':
                tmp_model = optimize_conv2d_batchnorm_block(tmp_model, model, input_layers, layer, next_layer, verbose)
                x = tmp_model.layers[-1].output
                skip_layers.append(output_layers[0])
                continue

            if layer_type in ['SeparableConv2D'] and next_layer_type == 'BatchNormalization':
                tmp_model = optimize_separableconv2d_batchnorm_block(tmp_model, model, input_layers, layer, next_layer, verbose)
                x = tmp_model.layers[-1].output
                skip_layers.append(output_layers[0])
                continue

        if x is None:
            input = get_copy_of_layer(layer, verbose)
            x = input
            tmp_model = Model(inputs=input.output, outputs=x.output)
        else:
            new_layer = get_copy_of_layer(layer, verbose)

            prev_layer = []
            for i in range(len(input_layers)):
                tens = tmp_model.get_layer(name=model.layers[input_layers[i]].name).output
                prev_layer.append(tens)
            if len(prev_layer) == 1:
                prev_layer = prev_layer[0]

            output_tensor, output_names = get_layers_without_output(tmp_model, verbose)
            x = new_layer(prev_layer)
            if layer.name not in output_names:
                output_tensor.append(x)
            else:
                output_tensor = x
            tmp_model = Model(inputs=input.output, outputs=output_tensor)
            tmp_model.get_layer(name=layer.name).set_weights(layer.get_weights())

    model = Model(inputs=input.output, outputs=x)
    return model
