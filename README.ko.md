# keras-contrib : Keras community contributions

[![Build Status](https://travis-ci.org/keras-team/keras-contrib.svg?branch=master)](https://travis-ci.org/keras-team/keras-contrib)

이 라이브러리는 파이썬 딥 러닝 라이브러리인 [케라스](http://www.keras.io)를 위한 레퍼지토리의 공식 확장 버전입니다. 추가적인 레이어, 액티베이션, 기능 줄이기, 최적화 기능 등과 같은 케라스 자체내에 포함되어 있지 않은 것들이 포함되어 있습니다. 이러한 모든 추가 모듈들은 코어 케라스와 모듈들의 코어로서 사용 될 수 있습니다.

Keras-Contrib의 커뮤니티 컨트리뷰션들이 시험되고 , 사용되고 , 타당성이 검증되며 그들의 유용성이 증명 된다면, 케라스 핵심 레퍼지토리로 병합될 수 있습니다. 케라스를 간결하고 깔끔하며, 정말로 간단하게 만드는 것에 대한 흥미만이 케라스에 대한 가장 유용한 컨트리뷰션을 만들어 내는 지름길 입니다. 컨트리뷰션 레퍼지토리는 새로운 기능을 위한 토대, 그리고 케라스 패러다임에 잘 맞지 않지만 유용한 기능을 위한 아카이브와 같은 두 기능을 수행 할 것입니다.

---
## Installation

#### Keras_team/Keras 를 위해 Keras_contrib를 설치해 주십시오. 
케라스를 어떻게 설치하는지에 대한 지시사항을 알고 싶다면 , [케라스 설치 페이지](https://keras.io/#installation)를 참고하십시오

```shell
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
python setup.py install
```

Pip를 사용하실 수도 있습니다

```shell
sudo pip install git+https://www.github.com/keras-team/keras-contrib.git
```

설치헤제 방법

```pip
pip uninstall keras_contrib
```

#### tensorflow.keras 를 위한 keras_contrib 설치하기

```shell
git clone https://www.github.com/keras-team/keras-contrib.git
cd keras-contrib
python convert_to_tf_keras.py
USE_TF_KERAS=1 python setup.py install
```

설치헤제 방법:

```shell
pip uninstall tf_keras_contrib
```

컨트리뷰트 가이드 라인을 확인하고 싶다면, [CONTRIBUTING.md](https://github.com/keras-team/keras-contrib/blob/master/CONTRIBUTING.md)
 를 확인해 주십시오.

 ---
 ## Example Usage

 Keras_Contrib에서 나온 모듈들은 Keras 그 자체내에서 사용 되던 모듈과 똑같이 사요됩니다

 ```python
 from keras.models import Sequential
 from keras.layers import Dense
 import numpy as np

 # I wish Keras had the Parametric Exponential Linear activation..
 # Oh, wait..!
 from keras_contrib.layers.advanced_activations import PELU

 # Create the Keras model, including the PELU advanced activation
 model = Sequential()
 model.add(Dense(100, input_shape=(10,)))
 model.add(PELU())

 # Compile and fit on random data
 model.compile(loss='mse', optimizer='adam')
 model.fit(x=np.random.random((100, 10)), y=np.random.random((100, 100)), epochs=5, verbose=0)

 # Save our model
 model.save('example.h5')
 ```

 ### A Common "Gotcha"

 As Keras-Contrib is external to the Keras core, loading a model requires a bit more work. While a pure Keras model is loadable with nothing more than an import of `keras.models.load_model`, a model which contains a contributed module requires an additional import of `keras_contrib`:

 ```python
 # Required, as usual
 from keras.models import load_model

 # Recommended method; requires knowledge of the underlying architecture of the model
 from keras_contrib.layers import PELU
 from keras_contrib.layers import GroupNormalization

 # Load our model
 custom_objects = {'PELU': PELU, 'GroupNormalization': GroupNormalization}
 model = load_model('example.h5', custom_objects)
 ```


