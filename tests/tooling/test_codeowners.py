import pytest
from github import Github
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

path_to_keras_contrib = pathlib.Path(__file__).resolve().parents[2]
path_to_codeowners = path_to_keras_contrib / 'CODEOWNERS'


def parse_codeowners():
    map_path_owner = []
    for line in open(path_to_codeowners, 'r'):
        line = line.strip()
        if line.startswith('#') or line == '':
            continue
        x = line.split(' ')
        path = path_to_keras_contrib / x[0]
        owner = x[-1]
        map_path_owner.append((path, owner))
    return map_path_owner


def test_codeowners_file_exist():
    for path, _ in parse_codeowners():
        assert path.exists()


def test_codeowners_user_exist():
    client = Github()
    for _, user in parse_codeowners():
        assert user[0] == '@'
        assert client.get_user(user[1:])


directories_to_test = [
    'examples',
    'keras_contrib/activations',
    'keras_contrib/applications',
    'keras_contrib/callbacks',
    'keras_contrib/constraints',
    'keras_contrib/datasets',
    'keras_contrib/initializers',
    'keras_contrib/layers',
    'keras_contrib/losses',
    'keras_contrib/metrics',
    'keras_contrib/optimizers',
    'keras_contrib/preprocessing',
    'keras_contrib/regularizers',
    'keras_contrib/wrappers'
]
directories_to_test = [path_to_keras_contrib / x for x in directories_to_test]

pattern_exclude = [
    '.gitkeep',
    '__init__.py'
]

# TODO: remove those files or find them owners.
exclude = [
    'examples/cifar10_clr.py',
    'examples/cifar10_densenet.py',
    'examples/cifar10_nasnet.py',
    'examples/cifar10_resnet.py',
    'examples/cifar10_ror.py',
    'examples/cifar10_wide_resnet.py',
    'examples/conll2000_chunking_crf.py',
    'examples/improved_wgan.py',
    'examples/jaccard_loss.py',
    'keras_contrib/activations/squash.py',
    'keras_contrib/callbacks/cyclical_learning_rate.py',
    'keras_contrib/callbacks/dead_relu_detector.py',
    'keras_contrib/callbacks/snapshot.py',
    'keras_contrib/applications/densenet.py',
    'keras_contrib/applications/nasnet.py',
    'keras_contrib/applications/resnet.py',
    'keras_contrib/applications/wide_resnet.py',
    'keras_contrib/constraints/clip.py',
    'keras_contrib/datasets/coco.py',
    'keras_contrib/datasets/conll2000.py',
    'keras_contrib/datasets/pascal_voc.py',
    'keras_contrib/initializers/convaware.py',
    'keras_contrib/losses/crf_losses.py',
    'keras_contrib/losses/dssim.py',
    'keras_contrib/losses/jaccard.py',
    'keras_contrib/layers/advanced_activations.py',
    'keras_contrib/layers/capsule.py',
    'keras_contrib/layers/convolutional.py',
    'keras_contrib/layers/core.py',
    'keras_contrib/layers/crf.py',
    'keras_contrib/layers/normalization.py',
    'keras_contrib/optimizers/ftml.py',
    'keras_contrib/optimizers/lars.py',
    'keras_contrib/optimizers/padam.py',
    'keras_contrib/optimizers/yogi.py',
    'keras_contrib/metrics/crf_accuracies.py',

]
exclude = [path_to_keras_contrib / x for x in exclude]


@pytest.mark.parametrize('directory', directories_to_test)
def test_all_files_have_owners(directory):
    files_with_owners = [x[0] for x in parse_codeowners()]
    for children in directory.iterdir():
        if children.is_file():
            if children.name in pattern_exclude:
                continue
            if children in exclude:
                continue
            assert children in files_with_owners


if __name__ == '__main__':
    pytest.main([__file__])
