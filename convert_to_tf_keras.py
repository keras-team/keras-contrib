import os


def replace_imports(file_path):
    if not file_path.endswith('.py') or file_path.endswith('convert_imports.py'):
        return
    with open(file_path, 'r') as f:
        text = f.read()

    # TODO: Find a way to remove this.
    # layer_test is not in the public API which is bad.
    text_updated = text.replace(
        'from keras.utils.test_utils import layer_test',
        'from tensorflow.python.keras.testing_utils import layer_test')

    text_updated = text_updated.replace('import keras', 'import tensorflow.keras')
    # we don't want to catch from keras_contrib.
    text_updated = text_updated.replace('from keras.', 'from tensorflow.keras.')
    text_updated = text_updated.replace('from keras ', 'from tensorflow.keras ')

    with open(file_path, 'w+') as f:
        f.write(text_updated)

    return text_updated != text


def main():
    """Run this function to convert the codebase to tf.keras"""
    nb_of_files_changed = 0
    for root, dirs, files in os.walk("."):
        for name in files:
            if replace_imports(os.path.join(root, name)):
                nb_of_files_changed += 1
    print('Chnaged imports in ' + str(nb_of_files_changed) + ' files.')


if __name__ == '__main__':
    main()
