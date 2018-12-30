import os


def replace_imports(file_path):
    if not file_path.endswith('.py') or file_path.endswith('setup.py'):
        return False
    with open(file_path, 'r') as f:
        text = f.read()

    # we don't want to catch 'from keras_contrib'.
    text_updated = text.replace('import keras.', 'import tensorflow.keras.')
    text_updated = text_updated.replace('import keras ', 'import tensorflow.keras ')
    text_updated = text_updated.replace('from keras.', 'from tensorflow.keras.')
    text_updated = text_updated.replace('from keras ', 'from tensorflow.keras ')

    with open(file_path, 'w+') as f:
        f.write(text_updated)

    return text_updated != text


def convert_to_tf_keras():
    """Run this function to convert the codebase to tf.keras"""
    nb_of_files_changed = 0
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for name in files:
            if replace_imports(os.path.join(root, name)):
                nb_of_files_changed += 1
    print('Chnaged imports in ' + str(nb_of_files_changed) + ' files.')
