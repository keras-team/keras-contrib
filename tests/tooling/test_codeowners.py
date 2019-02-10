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
        github_client = Github(os.environ['GITHUB_USER'], os.environ['GITHUB_PASSWORD'])
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


if __name__ == '__main__':
    pytest.main([__file__])
