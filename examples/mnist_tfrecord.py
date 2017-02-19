import tensorflow as tf

from keras.datasets import mnist
from keras import backend as K

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D
from keras.callbacks import EarlyStopping
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils

import keras_contrib.datasets.tfrecord as ktfr

import time
import numpy as np


sess = tf.Session()
K.set_session(sess)


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

def arch(inp):
  con1 = Convolution2D(32, 3, 3, border_mode='valid', activation = 'relu', subsample=(2,2))
  con2 = Convolution2D(32, 3, 3, activation = 'relu', subsample=(2,2))
  fla1 = Flatten()
  den1 = Dense(128, activation = 'relu')
  den2 = Dense(nb_classes, activation = 'softmax')
  out = den2(den1(fla1(con2(con1(inp)))))

  # fla1 = Flatten()
  # den1 = Dense(128, activation = 'relu')
  # den2 = Dense(128, activation = 'relu')
  # den3 = Dense(nb_classes, activation = 'softmax')
  # out = den3(den2(den1(fla1(inp))))

  return out 


ktfr.data_to_tfrecord(images=X_train, labels=y_train, filename='train.mnist.tfrecord')
# ktfr.data_to_tfrecord(images=X_test, labels=y_test, filename='test.mnist.tfrecord')

batch_size=32
nb_classes=10

x_train_, y_train_ = ktfr.read_and_decode('train.mnist.tfrecord', one_hot=True, n_class=nb_classes, is_train=True)

x_train_batch, y_train_batch = K.tf.train.shuffle_batch([x_train_, y_train_],
                                                batch_size=batch_size,
                                                capacity=2000,
                                                min_after_dequeue=1000,
                                                num_threads=32) # set the number of threads here

x_train_inp = Input(tensor=x_train_batch)
train_out = arch(x_train_inp)
train_model = Model(input=x_train_inp, output=train_out)
ktfr.compile_tfrecord(train_model, optimizer='rmsprop', loss='categorical_crossentropy', out_tensor_lst=[y_train_batch], metrics=['accuracy'])

train_model.summary()

ktfr.fit_tfrecord(train_model, X_train.shape[0], batch_size, nb_epoch=3)
train_model.save_weights('saved_wt.h5')



K.clear_session()



x_test_inp = Input(batch_shape=(None,)+(X_test.shape[1:]))
test_out = arch(x_test_inp)
test_model = Model(input=x_test_inp, output=test_out)
test_model.load_weights('saved_wt.h5')
test_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
test_model.summary()

loss, acc = test_model.evaluate(X_test, np_utils.to_categorical(y_test), nb_classes)
print '\nTest accuracy: {0}'.format(acc)


exit()


loss = tf.reduce_mean(categorical_crossentropy(y_train_batch, train_out))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

with sess.as_default():
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()

        _, loss_value = sess.run([train_op, loss], feed_dict={K.learning_phase(): 0})

        duration = time.time() - start_time

        if step % 100 == 0:
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                     duration))
        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
      coord.request_stop()

    coord.join(threads)
    sess.close()
