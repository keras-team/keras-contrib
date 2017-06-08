import sys
from os import path, environ
import os
import numpy as np
import zipfile
import time

try:
    # Python 3
    from urllib.request import urlretrieve
except:
    # Python 2
    from urllib import urlretrieve

APPNAME = "KerasContribGlove"
GLOVE_DL = "http://nlp.stanford.edu/data/glove.6B.zip"
FNAMES = ["glove.6B.%dd.txt" % size for size in [50, 100, 200, 300]]


# Show download rate, size, percentage, time passed
# https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
            (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


# Download embeddings if neccesary
def download_embeddings():
    if not path.exists(path.join(appdata,".downloaded")):
        print("Downloading pre-trained glove embeddings to %s" % (appdata))
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        urlretrieve(url, os.path.join(appdata, "glove.6B.zip"), reporthook)

        print("Unzipping glove embeddings.")
        zip_ref = zipfile.ZipFile(path.join(appdata,"glove.6B.zip"))
        zip_ref.extractall(appdata)
        zip_ref.close()

        print("Unzipping Complete: ",os.listdir(appdata),"were downloaded")
        open(path.join(appdata,".downloaded"),"a").close()
    else:
        print("Embedding files not found, but download directory already exists")


# Create appdata path
if sys.platform == 'win32':
    appdata = path.join(environ['APPDATA'], APPNAME)
else:
    appdata = path.expanduser(path.join("~", "." + APPNAME))

# Make appdata directory if doesn't exist
if not path.isdir(appdata):
    os.mkdir(appdata)

# Check if embedding files are present, download if not
for fname in FNAMES:
    if not os.path.exists(os.path.join(appdata, fname)):
        print("Missing embedding files in %s" % appdata)
        download_embeddings()
        break

if not path.isdir(appdata):
    print("Error, keras glove embeddings were not found in %s" % (appdata))
    raise FileNotFoundError

from keras.layers import Embedding
class GloveEmbedding(Embedding):
    def __init__(self,size,input_length,word_index,**kwargs):
        if not size in [50,100,200,300]:
            message = "Invalid Value %d passed as \"weights\" parameter.\n\tValid Values are: [50,100,200,300]"%num_weights
            raise ValueError(message)

        EMBED_SIZE = int(size)

        fname = "glove.6B.%dd.txt"%size
        fname = os.path.join(appdata,fname)

        embeddings_index = {}
        for line in open(fname):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embeddings_index[word] = coefs

        embedding_matrix=np.zeros((len(word_index)+1,EMBED_SIZE))
        for word,i in word_index.items():
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec

        return Embedding(
                len(word_index)+1,
                EMBED_SIZE,
                weights=[embedding_matrix],
                input_length=input_length,
                trainable=False,**kwargs)
