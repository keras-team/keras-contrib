import numpy as np
import json
import copy
import os
from scipy.sparse import *
import matplotlib
import matplotlib.pyplot as plt
import h5py


def get_wts(h5_file, layer_name, identifier):
    if h5_file.endswith(".h5") is not True:
        print("enter a valid h5 filename")
        return
    fh = h5py.File(h5_file, 'r')
    dataset_path = 'model_weights/' + layer_name + \
        '/' + layer_name + '/' + identifier + ':0'
    wts = np.asarray(fh[dataset_path])
    return wts


def plot_simil_mat(adj_mat):
    plt.imshow(adj_mat)
    plt.show(adj_mat)


def get_conn_comp_direct(wts, simil_thresh):
    adj_mat = get_simil_mat(wts, simil_thresh)
    ncomps, cc_list = get_conn_comp(adj_mat)
    return cc_list, adj_mat


def get_simil_mat(wts, simil_thresh):
    if len(wts.shape) == 4:
        wts = np.moveaxis(wts, 3, 0)
    elif len(wts.shape) == 3:
        wts = np.moveaxis(wts, 2, 0)
    else:
        print('input a valid 1D or 2D conv weight matrix')
        return
    wts_sqrt, wts_norm = norm_filter(wts, wts.shape[0])
    wts_norm_t = wts_norm.T
    covar = wts_norm.dot(wts_norm_t)
    adj_mat = np.greater(covar, simil_thresh)
    return adj_mat


def get_conn_comp(adj_mat):
    sz = adj_mat.shape
    n_comps, comp_labels = csgraph.connected_components(
        adj_mat, directed=False)
    cc_set = np.unique(comp_labels)

    # init a list of lists
    cc_list = []
    for i in range(len(cc_set)):
        cc_list.append([])

    # cc_list[i] stores the indices (j) of the filters in the i-th conn-comp
    for i in range(len(cc_set)):
        for j in range(sz[0]):
            if comp_labels[j] == cc_set[i]:
                cc_list[i].append(j)
    return n_comps, cc_list


def norm_filter(wts, no_filters):
    """
    normalization of a matrix (based on the L2 norm along axis=1)
    """
    wts = wts.reshape(no_filters, -1)
    # print("wts shape after reshape --> \n",wts)
    wts_sq = np.square(wts)
    # print("wts shape after square --> \n",wts_sq)
    wts_sum = np.sum(wts_sq, axis=1).reshape(wts.shape[0], 1)
    # print("wts shape after reshape, sq and sum axis = 1 --> \n",wts_sum)
    wts_sqrt = np.sqrt(wts_sum)
    wts_norm = wts / wts_sqrt
    # print("wts norm shape --> \n", wts_norm)

    return wts_sqrt, wts_norm


def scale_layers(wts, biases, next_weights):
    """
    scales current layer weights and biases and next layer activations
    """
    no_filters = wts.shape[0]
    wts_sqrt, wts_norm = norm_filter(wts, no_filters)
    b_norm = np.asarray([b / wts_sqrt[index]
                        for index, b in enumerate(biases)])
    layer_next_scaled_list = []
    for i in range(next_weights.shape[1]):
        layer_next_scaled_list.append(next_weights[:, i] * wts_sqrt[i])
    layer_next_scaled = np.asarray(layer_next_scaled_list)
    layer_next_scaled = np.moveaxis(layer_next_scaled, 1, 0)
    return (wts_norm, b_norm, layer_next_scaled)


def reduce_layers(layer_curr_weights,
                  layer_curr_biases, layer_next_scaled, cc_list):
    # reducing the weight filters
    layer_curr_filter_groups = [layer_curr_weights[x] for x in cc_list]
    layer_curr_filter_reduced = [x.mean(axis=0)
                                 for x in layer_curr_filter_groups]
    layer_curr_filter_reduced = np.asarray(layer_curr_filter_reduced)
    # layer_curr_filter_reduced = np.moveaxis(layer_curr_filter_reduced, 1, 0)
    print("new weight filter shape", layer_curr_filter_reduced.shape)

    # reducing biases
    layer_curr_bias_groups = [layer_curr_biases[x] for x in cc_list]
    layer_curr_bias_reduced = [x.mean() for x in layer_curr_bias_groups]
    layer_curr_bias_reduced = np.asarray(layer_curr_bias_reduced)
    print("new bias shape", layer_curr_bias_reduced.shape)

    # reducing next layer activations
    layer_next_groups = [layer_next_scaled[:, x] for x in cc_list]
    layer_next_reduced = [x.sum(axis=1) for x in layer_next_groups]
    layer_next_reduced = np.asarray(layer_next_reduced)
    # layer_next_reduced = np.moveaxis(layer_next_reduced, 1, 0)
    print("new next layer activations shape", layer_next_reduced.shape)

    return (layer_curr_filter_reduced, layer_curr_bias_reduced, layer_next_reduced)


def extract_current_and_next_weights(model, op_layer, next_layer):
    weights = model.get_layer(op_layer).get_weights()
    next_weights = model.get_layer(next_layer).get_weights()

    wts = weights[0]
    wts_shape = wts.shape
    biases = weights[1]

    next_wts = next_weights[0]
    next_biases = next_weights[1]

    # print("original shape of the weight matrix --> ",wts.shape)
    kernel_w = wts.shape[0]
    kernel_h = 1
    no_input_channels = wts.shape[1]
    no_filters = wts.shape[2]
    # print("required 4D tensor shapes (for 2D conv) --> ", no_filters,
    # no_input_channels, kernel_w, kernel_h)

    # arrange the weight matrix for the layer in shape (number_filters, ...)
    # wts = np.moveaxis(wts, 2, 0)

    # wts = wts.reshape(no_filters,no_input_channels,kernel_w,kernel_h)
    # print("wts shape before reshape --> ", wts.shape)

    return (wts, biases, next_wts, next_biases)


def modify_model_arch(json_mod_pack, model_mod_pack):
    """
    modifies the model json and h5 file with new parameters supplied in the json_mod_pack and model_mod_pack
    The model params in saved model h5 file are saved automatically in the provided model file name
    """
    json_file, current_layer, next_layer = json_mod_pack
    model_params_file, current_layer, next_layer, new_weights, new_biases, new_next_layer_weights, make_copy = model_mod_pack
    new_json = modify_json(json_file, current_layer, next_layer)
    new_params = modify_model_params(
        model_params_file,
        current_layer,
     next_layer,
     new_weights,
     new_biases,
     new_next_layer_weights,
     make_copy)
    return (new_json, new_params)


def modify_json(json_file, current_layer, next_layer, reduced_number_filters):
    """
    Function to modify the model json
    inputs:
    json_file: The current model json file
    current layer: target layer for reduction
    """
    with open(json_file) as f:
        new_model_serialized = json.load(f)

    new_model_serialized_copy = copy.deepcopy(new_model_serialized)

    for k, v in new_model_serialized_copy.items():
        if k == 'config':  # model config
                for m, n in v.items():
                        if m == 'layers':  # look for layers key
                            for l in n:  # layers indexed in the list - each is a dict
                                for layer_key, layer_val in l.items():
                                    if layer_key == 'config':
                                        for a, b in layer_val.items():
                                            if layer_val['name'] == current_layer and a == 'filters':
                                                layer_val[
                                                    a] = reduced_number_filters
                                                break
    # print(new_model_serialized_copy)
    new_model_json_name = 'modified_' + current_layer + '_' + json_file
    with open(new_model_json_name, 'w') as outfile:
        json.dump(new_model_serialized_copy, outfile)
    return new_model_json_name


def modify_model_params(model_params_file, current_layer, next_layer,
                        new_weights, new_biases, new_next_layer_weights, make_copy=0):
    """
    NOTE: This function does not save a copy of the h5 model params file, either make a copy and
    pass it to the function, or use make_copy=1. make_copy=0 by default

    Function to modify the model_params h5 file (keras based implementation works with
    both Theano and Tensorflow. After Nov 2017, Theano support from keras is fazed out.
    d
    Inputs:
    model_params_file: model h5 file
    current_layer: Target layer for reduction
    next_layer: Subsequent layer
    new_weights: (np array): new weights for the current layer. The size should conform to the keras h5 model save schema
    new_biases: (np array): new biases for the current layer. The size should conform to keras schema
    new_next_layer_weights: (np_array): new next layer weights. The size should conform to keras layer weights schema
    ---> sanity check: new_biases.shape[0] == new_weights.shape[0]

    current_layer: Name of the  Target layer for reduction
    next_layer: Name of the succeeding layer
    """
    op_h5file = model_params_file
    if make_copy == 1:
        new_model_params_file = 'modified_' + model_params_file
        h5repack_cmd = 'h5repack -i' + \
            model_params_file + '-o' + new_model_params_file
        os.system(h5repack_cmd)
        op_h5file = new_model_params_file
        print(
            "made a copy of the existing model params with the name {}".format(
                new_model_params_file))

    f1 = h5py.File(op_h5file, 'r+')
    path_dict = extract_h5_paths(current_layer, next_layer)
    existing_params = extract_existing_params(op_h5file, path_dict)
    if existing_params['wts'].shape == new_weights.shape:
        print(
            "Shape of the new weight matrix is same as the shape of current weight matrix. Use modify_sameShape_model_params() instead")
        return

    modify_h5_params(op_h5file, path_dict['wts'], new_weights)
    # print("current layer existing weights shape: ", existing_params['wts'].shape)
    # print("current layer new weights shape: ", )

    modify_h5_params(op_h5file, path_dict['biases'], new_biases)
    # print("current layer existing weights shape: ", existing_params['wts'].shape)
    # print("current layer new weights shape: ", )

    modify_h5_params(op_h5file, path_dict['next_wts'], new_next_layer_weights)
    # print("current layer existing weights shape: ", existing_params['wts'].shape)
    # print("current layer new weights shape: ", )

    modify_batchnorm(op_h5file, path_dict, new_biases.shape, existing_params)


def modify_sameShape_model_params(model_params_file, current_layer,
                                  next_layer, new_weights, new_biases, new_next_layer_weights, make_copy=0):
    op_h5file = model_params_file
    if make_copy == 1:
        new_model_params_file = model_params_file + 'modified_'
        h5repack_cmd = 'h5repack -i' + \
            model_params_file + '-o' + new_model_params_file
        os.system(h5repack_cmd)
        op_h5file = new_model_params_file

    f1 = h5py.File(new_model_params_file, 'r+')
    path_dict = extract_h5_paths(current_layer, next_layer)
    existing_params = extract_existing_params(op_h5file, path_dict)
    if existing_params['wts'].shape != new_weights.shape:
        print(
            "Shape of the new weight matrix is different from the shape of current weight matrix. Use modify_model_params() instead")
        return
    modify_h5_params(op_h5file, path_dict['wts'], new_weights)
    # print("current layer existing weights shape: ", existing_params['wts'].shape)
    # print("current layer new weights shape: ", )

    modify_h5_params(op_h5file, path_dict['biases'], new_biases)
    # print("current layer existing weights shape: ", existing_params['wts'].shape)
    # print("current layer new weights shape: ", )

    modify_h5_params(op_h5file, path_dict['next_wts'], new_next_layer_weights)
    # print("current layer existing weights shape: ", existing_params['wts'].shape)
    # print("current layer new weights shape: ", )

    modify_batchnorm(op_h5file, path_dict, new_biases.shape, existing_params)


def modify_batchnorm(op_h5file, path_dict, new_shape, existing_params):
    """
    modify the batchnorm layer in the h5 file
    """
    print('changing batchnorm layers to size', new_shape)
    bn_path_list = ['beta', 'gamma', 'moving_mean', 'moving_variance']
    fh = h5py.File(op_h5file, 'r+')
    for param in bn_path_list:
        path = path_dict[param]
        if path in fh.keys():
            print('modifying {}'.format(param))
            del fh[path]
            fh.create_dataset(
                path,
                data=existing_params[param][0:new_shape[0]])
        else:
            print("dataset doesn't exist at the provided path")
            return
    fh.close()
    return


def extract_h5_paths(current_layer, next_layer):
    """
    Given a layer name, extract the weights, biases, next layer weights and batchnorm params
    """
    path_dict = {}
    layer_idx = current_layer[-1]
    print("changing batchnorm for layer --> ", layer_idx)
    path_dict['wts'] = 'model_weights/' + \
        current_layer + '/' + current_layer + '/' + 'kernel:0'
    path_dict['biases'] = 'model_weights/' + \
        current_layer + '/' + current_layer + '/' + 'bias:0'
    path_dict['next_wts'] = 'model_weights/' + \
        next_layer + '/' + next_layer + '/' + 'kernel:0'
    path_dict['beta'] = 'model_weights/batch_normalization_' + \
        layer_idx + '/batch_normalization_' + layer_idx + '/' + 'beta:0'
    path_dict['gamma'] = 'model_weights/batch_normalization_' + \
        layer_idx + '/batch_normalization_' + layer_idx + '/' + 'gamma:0'
    path_dict['moving_mean'] = 'model_weights/batch_normalization_' + \
        layer_idx + '/batch_normalization_' + layer_idx + '/' + 'moving_mean:0'
    path_dict['moving_variance'] = 'model_weights/batch_normalization_' + \
        layer_idx + '/batch_normalization_' + \
        layer_idx + '/' + 'moving_variance:0'
    return path_dict


def extract_existing_params(oph5file, path_dict):
    """
    Returns layer params as numpy arrays.
    When using standalone:
    This function should be invoked after extract_h5_paths and the return value from there should be passed to extract_existing_params()
    """
    f = h5py.File(oph5file, 'r+')
    params_dict = {}
    params_dict['wts'] = np.asarray(f[path_dict['wts']])
    params_dict['baises'] = np.asarray(f[path_dict['biases']])
    params_dict['next_wts'] = np.asarray(f[path_dict['next_wts']])
    params_dict['beta'] = np.asarray(f[path_dict['beta']])
    params_dict['gamma'] = np.asarray(f[path_dict['gamma']])
    params_dict['moving_mean'] = np.asarray(f[path_dict['moving_mean']])
    params_dict['moving_variance'] = np.asarray(
        f[path_dict['moving_variance']])
    f.close()
    return params_dict


def modify_h5_params(op_h5file, path, new_weights):
    fh = h5py.File(op_h5file, 'r+')
    if path in fh.keys():
        del fh[path]
        fh.create_dataset(path, data=new_weights)
    else:
        print("dataset doesn't exist at the provided path")
        return
    fh.close()
    return
