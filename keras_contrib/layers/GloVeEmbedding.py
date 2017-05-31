import sys
from os import path, environ
import os
import numpy as np
import errno

#Get an appdata folder for our module.
APPNAME = "KerasContribGlove"
appdata = ""
ENDC = '\033[0m'

if sys.platform == 'win32':
    appdata = path.join(environ['APPDATA'], APPNAME)
else:
    appdata = path.expanduser(path.join("~", "." + APPNAME))

if not os.path.exists(appdata) and not path.isdir(appdata):
    try:
        os.makedirs(appdata)
    except:
        raise

if not path.exists(path.join(appdata,".downloaded")):
    print("\033[94mDownloading pre-trained glove embeddings to %s" % (appdata),ENDC)
    print("\033[93mWarning, this might take awhile.  Thank you for your patience.",ENDC)

    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    import requests
    r = requests.get(url, stream=True)

    with open(path.join(appdata,"glove.6B.zip"), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    print("\033[94mUnzipping glove embeddings.",ENDC)
    import zipfile
    zip_ref = zipfile.ZipFile(path.join(appdata,"glove.6B.zip"))
    zip_ref.extractall(appdata)
    zip_ref.close()
    print("\033[94mUnzipping Complete: ",os.listdir(appdata),"were downloaded",ENDC)
    open(path.join(appdata,".downloaded"),"a").close()

"""
TODO:// Allow users to download GloVe embeddings trained on different dataset
"""
def GloVeEmbedding(size,input_length,word_index,**kwargs):
    from keras.layers import Embedding
    if not size in [50,100,200,300]:
        message = "Invalid Value %d passed as \"weights\" parameter.\n\tValid Values are: [50,100,200,300]"%num_weights
        raise ValueError(message)

    EMBED_SIZE = int(size)

    fname = "glove.6B.%dd.txt"%size
    fname = os.path.join(appdata,fname)

    # Credit: Fchollet for this section

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
