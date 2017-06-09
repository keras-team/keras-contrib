from keras_contrib.wrappers import GloveEmbedding
from keras.models import Sequential

# Use embedding size of 300
EMBED_SIZE = 300
MAX_TEXT_LEN = 10
word_index = {
    "hello": 0,
    "world": 1,
    "foo": 2,
    "bar": 3
}
# Add into a model to ensure comtability with models/
rnn = Sequential()
rnn.add(GloveEmbedding(EMBED_SIZE,
                       MAX_TEXT_LEN,
                       word_index))
rnn.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

# Ensure proper output shapes.
assert(rnn.layers[-1].compute_output_shape((None, 10)) == (None, 10, 300))
