import os
import sys

list_conversions = [('import keras.', 'import tensorflow.keras.'),
                    ('import keras ', 'from tensorflow import keras '),
                    ('import keras\n', 'from tensorflow import keras\n'),
                    ('from keras.', 'from tensorflow.keras.'),
                    ('from keras ', 'from tensorflow.keras ')]


def replace_imports_in_text(string, revert):
    if revert:
        list_imports_to_change = [x[::-1] for x in list_conversions]
    else:
        list_imports_to_change = list_conversions

    text_updated = string
    for old_str, new_str in list_imports_to_change:
        text_updated = text_updated.replace(old_str, new_str)
    return text_updated


def replace_imports_in_file(file_path, revert):
    if not file_path.endswith('.py'):
        return False
    if os.path.abspath(file_path) == os.path.abspath(__file__):
        return False
    with open(file_path, 'r') as f:
        text = f.read()

    text_updated = replace_imports_in_text(text, revert)

    with open(file_path, 'w+') as f:
        f.write(text_updated)

    return text_updated != text


def convert_codebase(revert):
    nb_of_files_changed = 0
    keras_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(keras_dir):
        for name in files:
            if replace_imports_in_file(os.path.join(root, name), revert):
                nb_of_files_changed += 1
    print('Changed imports in ' + str(nb_of_files_changed) + ' files.')
    print('Those files were found in the directory ' + keras_dir)


def convert_to_tf_keras():
    """Convert the codebase to tf.keras"""
    convert_codebase(False)


def convert_to_keras_team_keras():
    """Convert the codebase from tf.keras to keras-team/keras"""
    convert_codebase(True)


def test_replace_imports():
    python_code = """
    import keras
    from keras import backend as K
    import os
    import keras_contrib
    import keras_contrib.layers as lay
    import keras.layers
    from keras.layers import Dense

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        function = tf.max
    """

    expected_code = """
    from tensorflow import keras
    from tensorflow.keras import backend as K
    import os
    import keras_contrib
    import keras_contrib.layers as lay
    import tensorflow.keras.layers
    from tensorflow.keras.layers import Dense

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        function = tf.max
    """

    code_with_replacement = replace_imports_in_text(python_code, False)
    assert expected_code == code_with_replacement
    assert python_code == replace_imports_in_text(code_with_replacement, True)


if __name__ == '__main__':
    if '--revert' in sys.argv:
        convert_to_keras_team_keras()
    else:
        convert_to_tf_keras()
