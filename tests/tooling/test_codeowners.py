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


if __name__ == '__main__':
    pytest.main([__file__])
