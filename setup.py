from setuptools import setup
from setuptools import find_packages
import sys
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


if "--use-tf-keras" in sys.argv:
    sys.argv.remove("--use-tf-keras")
    name = 'tf_keras_contrib'
    install_requires = []
    convert_to_tf_keras()
else:
    name = 'keras_contrib'
    install_requires = ['keras']

setup(name=name,
      version='2.0.8',
      description='Keras Deep Learning for Python, Community Contributions',
      author='Fariz Rahman',
      author_email='farizrahman4u@gmail.com',
      url='https://github.com/farizrahman4u/keras-contrib',
      license='MIT',
      install_requires=install_requires,
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot>=1.2.0'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
