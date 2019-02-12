import os
import pytest
from github import Github
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

path_to_keras_contrib = pathlib.Path(__file__).resolve().parents[2]
path_to_codeowners = path_to_keras_contrib / 'CODEOWNERS'

authenticated = True
try:
    github_client = Github(os.environ['GITHUB_TOKEN'])
except KeyError:
    try:
        github_client = Github(os.environ['GITHUB_USER'],
                               os.environ['GITHUB_PASSWORD'])
    except KeyError:
        authenticated = False


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


@pytest.mark.skipif(not authenticated,
                    reason='It should be possible to run the test without'
                           'authentication, but we might get our request refused'
                           'by github. To be deterministic, we\'ll disable it.')
def test_codeowners_user_exist():
    for _, user in parse_codeowners():
        assert user[0] == '@'
        assert github_client.get_user(user[1:])


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
    'keras_contrib/callbacks/cyclical_learning_rate.py',
    'keras_contrib/callbacks/dead_relu_detector.py',
    'keras_contrib/applications/resnet.py',
    'keras_contrib/constraints/clip.py',
    'keras_contrib/datasets/coco.py',
    'keras_contrib/datasets/conll2000.py',
    'keras_contrib/datasets/pascal_voc.py',
    'keras_contrib/initializers/convaware.py',
    'keras_contrib/losses/crf_losses.py',
    'keras_contrib/losses/dssim.py',
    'keras_contrib/losses/jaccard.py',
    'keras_contrib/layers/advanced_activations/pelu.py',
    'keras_contrib/layers/advanced_activations/srelu.py',
    'keras_contrib/layers/convolutional/cosineconvolution2d.py',
    'keras_contrib/layers/core.py',
    'keras_contrib/layers/crf.py',
    'keras_contrib/layers/normalization/instancenormalization.py',
    'keras_contrib/optimizers/ftml.py',
    'keras_contrib/optimizers/lars.py',
    'keras_contrib/metrics/crf_accuracies.py',
]
exclude = [path_to_keras_contrib / x for x in exclude]


@pytest.mark.parametrize('directory', directories_to_test)
def test_all_files_have_owners(directory):
    files_with_owners = [x[0] for x in parse_codeowners()]
    for root, dirs, files in os.walk(directory):
        for name in files:
            file_path = pathlib.Path(root) / name
            if file_path.suffix != '.py':
                continue
            if file_path.name == '__init__.py':
                continue
            if file_path in exclude:
                continue
            assert file_path in files_with_owners


if __name__ == '__main__':
    pytest.main([__file__])
