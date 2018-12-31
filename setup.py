from setuptools import setup
from setuptools import find_packages
import os


if os.environ.get('USE_TF_KERAS', None) == '1':
    name = 'tf_keras_contrib'
    install_requires = []
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
