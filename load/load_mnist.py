# coding=UTF-8

import os
import os.path
import urllib.request
import gzip
import pickle
import numpy as np

mnist_url = "http://yann.lecun.com/exdb/mnist/"
file_dic = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"}

save_dir = os.path.dirname(os.path.abspath(__file__)) + "/data"
save_name = save_dir + "/mnist.pkl"


def download():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for key, file_name in file_dic.items():
        if os.path.exists(save_dir + "/" + file_name):
            print("{} is already exist.".format(key))
            continue
        else:
            print("download {}".format(key))
            urllib.request.urlretrieve(mnist_url + file_name, save_dir + "/" + file_name)


def make_pickle():
    if os.path.exists(save_name):
        print("pickle file is already exist")
        return
    dataset = {}
    for key, file_name in file_dic.items():
        file_path = save_dir + "/" + file_name
        with gzip.open(file_path, 'rb') as f:
            dt = np.dtype('>I')
            magic_num = np.frombuffer(f.read(), dt, count=1, offset=0)
            print("magic_num {}".format(magic_num))
        with gzip.open(file_path, 'rb') as f:
            if magic_num == 2049:
                labels = np.frombuffer(f.read(), np.uint8, offset=8)
                dataset[key] = labels
            elif magic_num == 2051:
                images = np.frombuffer(f.read(), np.uint8, offset=16)
                images = images.reshape(-1, 784)
                dataset[key] = images
    with open(save_name, 'wb') as f:
        print("make pickle file")
        pickle.dump(dataset, f, -1)


def load():
    if not os.path.exists(save_name):
        download()
        make_pickle()
    with open(save_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


if __name__ == "__main__":
    download()
    make_pickle()
    dataset = load()
    for key in file_dic.keys():
        print(dataset[key][:5])
