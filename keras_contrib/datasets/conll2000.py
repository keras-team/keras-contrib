import numpy
from keras.utils.data_utils import get_file
from zipfile import ZipFile
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import cifar10


def load_data(path='conll2000.zip', min_freq=2):
    path = get_file(path, origin='https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/conll2000.zip')
    print path
    archive = ZipFile(path, 'r')
    train = _parse_data(archive.open('conll2000/train.txt'))
    test = _parse_data(archive.open('conll2000/test.txt'))
    archive.close()

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = ['<pad>', '<unk>'] + [w for w, f in word_counts.iteritems() if f >= min_freq]
    pos_tags = sorted(list(set(row[1] for sample in train + test for row in sample)))  # in alphabetic order
    chunk_tags = sorted(list(set(row[2] for sample in train + test for row in sample)))  # in alphabetic order

    train = _process_data(train, vocab, pos_tags, chunk_tags)
    test = _process_data(test, vocab, pos_tags, chunk_tags)
    return train, test, (vocab, pos_tags, chunk_tags)


def _parse_data(fh):
    string = fh.read()
    data = [[row.split() for row in sample.split('\n')] for sample in string.strip().split('\n\n')]
    fh.close()
    return data


def _process_data(data, vocab, pos_tags, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_pos = [[pos_tags.index(w[1]) for w in s] for s in data]
    y_chunk = [[chunk_tags.index(w[2]) for w in s] for s in data]

    x = pad_sequences(x, maxlen)  # left padding

    y_pos = pad_sequences(y_pos, maxlen, value=-1)  # lef padded with -1. Indeed, any interger works as it will be masked
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_pos = numpy.eye(len(pos_tags), dtype='float32')[y]
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y]
    else:
        y_pos = numpy.expand_dims(y_pos, 2)
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_pos, y_chunk
