"""
Downloads the following:
- Stanford parser
- Stanford POS tagger
- Glove vectors
- SICK dataset (semantic relatedness task)
- Stanford Sentiment Treebank (sentiment classification task)
"""

from __future__ import print_function
import urllib2
import sys
import os
import shutil
import zipfile
import gzip
from helpers import io_utils

def download(url, dirpath):
    io_utils.check_dir(dirpath)
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    try:
        u = urllib2.urlopen(url)
    except:
        print("URL %s failed to open" %url)
        raise Exception
    try:
        f = open(filepath, 'wb')
    except:
        print("Cannot write %s" %filepath)
        raise Exception
    try:
        filesize = int(u.info().getheaders("Content-Length")[0])
    except:
        print("URL %s failed to report length" %url)
        raise Exception
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_wordvecs(dirpath):
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))

def create_glove_vocab(dirpath):
    glove_path = os.path.join(dirpath, 'glove.840B.300d.txt')
    with open(glove_path) as f:
        line = f.readline().split(' ')
        word = line[0]
        vecs = map(lambda item: float(item), line[1:])
        print(word)
        print(vecs)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # data
    data_dir = os.path.join(base_dir, 'word_embeddings')
    wordvec_dir = os.path.join(data_dir, 'glove')

    # download dependencies
    download_wordvecs(wordvec_dir)

    # create the vocabulary file from the word embeddings
    create_glove_vocab(wordvec_dir)