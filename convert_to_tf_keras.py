import os


def replace_imports(file_path):
    if not file_path.endswith('.py') or file_path.endswith('convert_imports.py'):
        return
    with open(file_path, 'r') as f:
        text = f.read()

    text = text.replace('import keras', 'import tensorflow.keras')
    text = text.replace('from keras import', 'from tensorflow.keras import')

    with open(file_path, 'w+') as f:
        f.write(text)


def main():
    """Run this function to convert the codebase to tf.keras"""
    for root, dirs, files in os.walk("."):
        for name in files:
            replace_imports(os.path.join(root, name))


if __name__ == '__main__':
    main()
