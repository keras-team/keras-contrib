from setuptools import setup
from setuptools import find_packages


setup(name='keras_contrib',
      version='1.2.1',
      description='Keras community contributions',
      author='Fariz Rahman',
      author_email='farizrahman4u@gmail.com',
      url='https://github.com/farizrahman4u/keras-contrib',
      license='MIT',
      install_requires=['keras'],
      packages=find_packages())
