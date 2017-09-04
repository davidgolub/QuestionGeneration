sudo pip uninstall tensorflow
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl

sudo pip install --upgrade pip
sudo pip install --upgrade $TF_BINARY_URL

sudo pip install requests
sudo pip install tqdm
sudo pip install pandas
sudo pip install nltk

sudo apt-get update
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:git-core/ppa
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install