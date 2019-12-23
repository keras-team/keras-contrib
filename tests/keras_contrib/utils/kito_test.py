import pytest
import numpy as np
import time
from keras_contrib.utils.kito import reduce_keras_model


def compare_two_models_results(m1, m2, test_number=10000, max_batch=10000):
    i1 = m1.layers[0].input_shape
    i2 = m2.layers[0].input_shape
    if i1 != i2:
        print('Different input shapes for models {} vs {}'.format(i1, i2))
    o1 = m1.layers[-1].output_shape
    o2 = m2.layers[-1].output_shape
    if o1 != o2:
        print('Different output shapes for models {} vs {}'.format(o1, o2))

    t1 = 0
    t2 = 0
    max_error = 0.0
    avg_error = 0.0
    count = 0
    for i in range(0, test_number, max_batch):
        tst = min(test_number - i, max_batch)
        print('Generate random images {}...'.format(tst))
        matrix = np.random.uniform(0.0, 1.0, (tst,) + i1[1:])
        print(matrix.shape)

        start_time = time.time()
        res1 = m1.predict(matrix)
        t1 += time.time() - start_time

        start_time = time.time()
        res2 = m2.predict(matrix)
        t2 += time.time() - start_time

        abs_diff = np.abs(res1 - res2)
        max_error = max(max_error, abs_diff.max())
        avg_error += abs_diff.sum()
        count += abs_diff.size

    print("Initial model prediction time for {} random images: {:.2f} seconds".format(test_number, t1))
    print("Reduced model prediction time for {} same random images: {:.2f} seconds".format(test_number, t2))
    print('Models raw max difference: {} (Avg difference: {})'.format(max_error, avg_error / count))
    return max_error


def get_custom_multi_io_model():
    from keras.layers import Input, Conv2D, BatchNormalization, Activation
    from keras.layers import Concatenate, GlobalAveragePooling2D, Dense
    from keras.models import Model

    inp1 = Input((32, 32, 3))
    inp2 = Input((32, 32, 3))

    branch1 = Conv2D(32, (3, 3), kernel_initializer='random_uniform')(inp1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)

    branch2 = Conv2D(32, (3, 3), kernel_initializer='random_uniform')(inp2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)

    x = Concatenate(axis=-1, name='concat')([branch1, branch2])

    branch3 = Conv2D(32, (3, 3), kernel_initializer='random_uniform')(x)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation('relu')(branch3)

    out1 = GlobalAveragePooling2D()(branch2)
    out1 = Dense(1, activation='sigmoid', name='fc1')(out1)

    out2 = GlobalAveragePooling2D()(branch3)
    out2 = Dense(1, activation='sigmoid', name='fc2')(out2)

    custom_model = Model(inputs=[inp1, inp2], outputs=[out1, out2])
    return custom_model


def get_simple_submodel():
    from keras.layers import Input, Conv2D
    from keras.models import Model
    inp = Input((28, 28, 4))
    branch = Conv2D(8, (3, 3), padding='same', kernel_initializer='random_uniform')(inp)
    model = Model(inputs=inp, outputs=branch)
    return model


def get_custom_model_with_other_model_as_layer():
    from keras.layers import Input, Conv2D
    from keras.layers import Concatenate
    from keras.models import Model

    inp1 = Input((28, 28, 3))
    branch1 = Conv2D(4, (3, 3), padding='same', kernel_initializer='random_uniform')(inp1)
    branch2 = Conv2D(4, (3, 3), padding='same', kernel_initializer='random_uniform')(inp1)
    m = get_simple_submodel()
    x1 = m(branch1)
    x2 = m(branch2)
    x = Concatenate(axis=-1, name='concat')([x1, x2])
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='random_uniform')(x)
    custom_model = Model(inputs=inp1, outputs=x)
    return custom_model


def get_neural_net(type):
    model = None
    if type == 'mobilenet_small':
        from keras.applications.mobilenet import MobileNet
        model = MobileNet((128, 128, 3), depth_multiplier=1, alpha=0.25, include_top=True, weights='imagenet')
    elif type == 'mobilenet':
        from keras.applications.mobilenet import MobileNet
        model = MobileNet((224, 224, 3), depth_multiplier=1, alpha=1.0, include_top=True, weights='imagenet')
    elif type == 'resnet50':
        from keras.applications.resnet50 import ResNet50
        model = ResNet50(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    elif type == 'inception_v3':
        from keras.applications.inception_v3 import InceptionV3
        model = InceptionV3(input_shape=(299, 299, 3), include_top=True, weights='imagenet')
    elif type == 'inception_resnet_v2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        model = InceptionResNetV2(input_shape=(299, 299, 3), include_top=True, weights='imagenet')
    elif type == 'xception':
        from keras.applications.xception import Xception
        model = Xception(input_shape=(299, 299, 3), include_top=True, weights='imagenet')
    elif type == 'densenet121':
        from keras.applications.densenet import DenseNet121
        model = DenseNet121(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    elif type == 'densenet169':
        from keras.applications.densenet import DenseNet169
        model = DenseNet169(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    elif type == 'densenet201':
        from keras.applications.densenet import DenseNet201
        model = DenseNet201(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    elif type == 'nasnetmobile':
        from keras.applications.nasnet import NASNetMobile
        model = NASNetMobile(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
    elif type == 'nasnetlarge':
        from keras.applications.nasnet import NASNetLarge
        model = NASNetLarge(input_shape=(331, 331, 3), include_top=True, weights='imagenet')
    elif type == 'vgg16':
        from keras.applications.vgg16 import VGG16
        model = VGG16(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')
    elif type == 'vgg19':
        from keras.applications.vgg19 import VGG19
        model = VGG19(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')
    elif type == 'multi_io':
        model = get_custom_multi_io_model()
    elif type == 'multi_model_layer':
        model = get_custom_model_with_other_model_as_layer()
    return model


def test_kito_NN_Reduction():
    import keras.backend as K
    models_to_test = ['mobilenet_small', 'mobilenet', 'resnet50', 'inception_v3', 'inception_resnet_v2', 'xception',
                      'densenet121', 'densenet169', 'densenet201',
                      'nasnetmobile', 'nasnetlarge', 'multi_io', 'multi_model_layer']
    # Comment line below for full model testing
    models_to_test = ['mobilenet_small', 'multi_io', 'multi_model_layer']

    for model_name in models_to_test:
        print('Go for: {}'.format(model_name))
        model = get_neural_net(model_name)
        print(model.summary())
        start_time = time.time()
        model_reduced = reduce_keras_model(model, verbose=True)
        print("Reduction time: {:.2f} seconds".format(time.time() - start_time))
        print(model_reduced.summary())
        print('Initial model number layers: {}'.format(len(model.layers)))
        print('Reduced model number layers: {}'.format(len(model_reduced.layers)))
        print('Compare models...')
        if model_name in ['nasnetlarge', 'deeplab_v3plus_mobile', 'deeplab_v3plus_xception']:
            max_error = compare_two_models_results(model, model_reduced, test_number=10000, max_batch=128)
        elif model_name in ['mobilenet_small', 'multi_io', 'multi_model_layer']:
            max_error = compare_two_models_results(model, model_reduced, test_number=1000, max_batch=1000)
        else:
            max_error = compare_two_models_results(model, model_reduced, test_number=10000, max_batch=10000)
        # K.clear_session()
        assert max_error < 1e-04


if __name__ == '__main__':
    pytest.main([__file__])
