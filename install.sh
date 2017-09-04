#!/bin/bash
pip install tqdm
pip install unidecode
pip install textblob
pip3 install tqdm 
pip3 install unidecode 
pip3 install textblob
pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
sudo pip3 install --upgrade pip
sudo pip3 install --upgrade $TF_BINARY_URL
sudo pip install tensorflow-gpu
pip install spacy && python -m spacy download en
pip3 install spacy && python3 -m spacy download en
cd bidaf
./download.sh
cd ../





