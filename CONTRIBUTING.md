# On Github Issues and Pull Requests

Found a bug? Want to contribute changes to the codebase? Make sure to read this first.

## Update Your Environment

To easily update Keras: `pip install git+https://www.github.com/keras-team/keras.git --upgrade`

To easily update Keras-Contrib: `pip install git+https://www.github.com/keras-team/keras-contrib.git --upgrade`

To easily update Theano: `pip install git+git://github.com/Theano/Theano.git --upgrade`

To update TensorFlow: See [TensorFlow Installation instructions](https://github.com/tensorflow/tensorflow#installation)

## Bug reporting

Your code doesn't work, **and you have determined that the issue lies with Keras-Contrib**? Follow these steps to report a bug.

1. Your bug may already be fixed. Make sure to update to the current Keras master branch and Keras-Contrib master branch, as well as the latest Theano/TensorFlow master branch.

2. [Search for similar issues](https://github.com/keras-team/keras-contrib/issues?utf8=%E2%9C%93&q=is%3Aissue). It's possible somebody has encountered this bug already. Still having a problem? Open an issue on Github to let us know.

3. Make sure you provide us with useful information about your configuration: what OS are you using? What Keras backend are you using? Are you running on GPU? If so, what is your version of Cuda, of cuDNN? What is your GPU?

4. Provide us with a script to reproduce the issue. This script should be runnable as-is and should not require external data download (use randomly generated data if you need to run a model on some test data). We recommend that you use Github Gists to post your code. Any issue that cannot be reproduced is likely to be closed.

5. If possible, take a stab at fixing the bug yourself --if you can!

The more information you provide, the easier it is for us to validate that there is a bug and the faster we'll be able to take action. If you want your issue to be resolved quickly, following the steps above is crucial.

## Pull Requests

We love pull requests. Here's a quick guide:

1. If your PR introduces a change in functionality, make sure you start by opening an issue to discuss whether the change should be made, and how to handle it. This will save you from having your PR closed down the road! Of course, if your PR is a simple bug fix, you don't need to do that.

2. Ensure that your environment (Keras, Keras-Contrib, and your backend) are up to date. See "Update Your Environment". Create a new branch for your changes.

3. Write the code. This is the hard part! If you are adding a layer, advanced activation, or any other feature which has configurable parameters, please ensure that the feature is searializeable (to allow for saving and loading). For details on this aspect, please see the Keras ["Writing Your Own Layer"](https://keras.io/layers/writing-your-own-keras-layers/) guide and the source code for the relevant feature type from both Keras and Keras-Contrib.

4. Make sure any new function or class you introduce has proper docstrings. Make sure any code you touch still has up-to-date docstrings and documentation.

5. Write tests. Your code should have full unit test coverage. If you want to see your PR merged promptly, this is crucial. If your PR is a bug fix, it is advisable to add a new test, which, without your fix in this PR, would have failed.

6. Run our test suite locally. It's easy: within the root Keras-Contrib folder, simply run: `py.test tests/`.
  - You will need to install `pytest`, `coveralls`, `pytest-cov`, `pytest-xdist`: `pip install pytest pytest-cov python-coveralls pytest-xdist pep8 pytest-pep8`

7. Make sure all tests are passing:
  - with the Theano backend, on Python 2.7 and Python 3.5
  - with the TensorFlow backend, on Python 2.7
  - **Please Note:** all tests run on top of the very latest Keras master branch.

8. We use PEP8 syntax conventions, but we aren't dogmatic when it comes to line length. Make sure your lines stay reasonably sized, though. To make your life easier, we recommend running a PEP8 linter:
  - Install PEP8 packages: `pip install pep8 pytest-pep8 autopep8`
  - Run a standalone PEP8 check: `py.test --pep8 -m pep8`
  - You can automatically fix some PEP8 error by running: `autopep8 -i --select <errors> <FILENAME>` for example: `autopep8 -i --select E128 tests/keras/backend/test_backends.py`

9. When committing, use appropriate, descriptive commit messages. Make sure that your branch history is not a string of "bug fix", "fix", "oops", etc. When submitting your PR, squash your commits into a single commit with an appropriate commit message, to make sure the project history stays clean and readable. See ['rebase and squash'](http://rebaseandsqua.sh/) for technical help on how to squash your commits.

10. Update the documentation. If introducing new functionality, make sure you include code snippets demonstrating the usage of your new feature.

11. Submit your PR. If your changes have been approved in a previous discussion, and if you have complete (and passing) unit tests, your PR is likely to be merged promptly. Otherwise, well...

## About keras-team/keras and tensorflow.keras

This repo supports both keras-team/keras and tensorflow.keras. The way this is done is by changing all the imports in the code by parsing it. This is checked with travis.ci every time you push a commit in a pull request. 

There are a number of reasons why your code would work with keras-team/keras but not with tf.keras. The most common is that you use keras' private API. Since both keras are only similar in behavior with respect to their public API, you should only use this. Otherwise it's likely that the function you are using is not in the same place in tf.keras (or does not even exist at all).

Another gotcha is that when creating custom layers and implementing the `build` function, keras-team/keras expects as `input_shape` a tuple of ints. With tf.keras, `input_shape` is a tuple with `Dimensions` objects. This is likely to make the code incompatible. To solve this problem, you should do:

```python
from keras.layers import Layer
from keras_contrib.utils.test_utils import to_tuple


class MyLayer(Layer):
    ...
    
    def build(self, input_shape):
        input_shape = to_tuple(input_shape)
        # now `input_shape` is a tuple of ints or None like in keras-team/keras
        ...
```

To change all the imports in your code to tf.keras to test compatibility, you can do:
```
python convert_to_tf_keras.py
```

To convert your codebase back to keras-team/keras, do:
```
python convert_to_tf_keras.py --revert
```

Note that you are strongly encouraged to commit your code before in case the parsing would go wrong. To discard all the changes you made since the previous commit:
```
# saves a copy of your current codebase in the git stash and comes back to the previous commit
git stash

git stash pop # get your copy back from the git stash if you need to.
```

## A Note for Contributors

Both Keras-Contrib and Keras operate under the [MIT License](LICENSE). At the discretion of the maintainers of both repositories, code may be moved from Keras-Contrib to Keras and vice versa.

The maintainers will ensure that the proper chain of commits will flow in both directions, with proper attribution of code. Maintainers will also do their best to notify contributors when their work is moved between repositories.

## About the `CODEOWNERS` file

If you add a new feature to keras-contrib, you should add yourself and your file in the `CODEOWNERS` file. Doing so will, in the future, tag you whenever an issue or a pull request about your feature is opened. Be aware that it represents some work, and in addition of being tagged, we would appreciate that you review new pull requests related to your feature.
