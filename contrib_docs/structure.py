from keras_contrib import layers
from keras_contrib.layers import advanced_activations
from keras_contrib import initializers
from keras_contrib import optimizers
from keras_contrib import callbacks
from keras_contrib import losses
from keras_contrib import backend
from keras_contrib import constraints


EXCLUDE = {
    'Optimizer',
    'TFOptimizer',
    'Wrapper',
    'get_session',
    'set_session',
    'CallbackList',
    'serialize',
    'deserialize',
    'get',
    'set_image_dim_ordering',
    'normalize_data_format',
    'image_dim_ordering',
    'get_variable_shape',
    'Constraint'
}


# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
PAGES = [
    {
        'page': 'layers/core.md',
        'classes': [
            layers.CosineDense,
        ],
    },
    {
        'page': 'layers/convolutional.md',
        'classes': [
            layers.CosineConv2D,
            layers.SubPixelUpscaling,
        ],
    },
    {
        'page': 'layers/normalization.md',
        'classes': [
            layers.InstanceNormalization,
            layers.GroupNormalization
        ],
    },
    {
        'page': 'layers/advanced-activations.md',
        'classes': [
            layers.SineReLU,
            layers.SReLU,
            layers.Swish,
            layers.PELU
        ],
    },
    {
        'page': 'layers/crf.md',
        'classes': [
            layers.CRF,
        ]
    },
    {
        'page': 'losses.md',
        'all_module_functions': [losses],
    },
    {
        'page': 'initializers.md',
        'all_module_classes': [initializers],
    },
    {
        'page': 'optimizers.md',
        'all_module_classes': [optimizers],
    },
    {
        'page': 'callbacks.md',
        'all_module_classes': [callbacks],
    },
    {
        'page': 'backend.md',
        'all_module_functions': [backend],
    },
    {
        'page': 'constraints.md',
        'all_module_classes': [constraints],
    },
]

ROOT = 'http://keras.io/'
