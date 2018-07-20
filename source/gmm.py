# coding=UTF-8
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from PIL import Image
from load.load_mnist import load
from source.k_means import k_means, mask_one_hot, draw_heatmap, entropy_score
from sklearn.datasets import load_iris


def gaussian(x, mu, sigma):
    d = len(mu)

    dev = np.linalg.det(sigma) ** .5
    coefficient = 1./(2. * np.pi) ** (d * .5) / dev
    return coefficient * np.exp(-.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu))


def e_step(data, pi, mu, sigma, K):
    burden_rate = np.array([pi[k] * np.diag(gaussian(data, mu[:, k, np.newaxis], sigma[k])) for k in range(K)])

    # N = n(data, mu, sigma)  # 60000 * 60000
    # burden_rate = pi * N / np.sum(pi * N)  # 10 * 60000

    return burden_rate.T


def m_step(data, burden_rate, K):
    s = np.sum(burden_rate, axis=0)
    pi = 1/len(data) * s  # shape: 10
    mu = data @ burden_rate / s  # shape: 784 * 10

    size = len(data[:, 0])
    sigma = np.zeros((K, size, size))
    for k in range(K):
        post = (data - mu[:, k, np.newaxis]).T[:, :, np.newaxis] @ (data - mu[:, k, np.newaxis]).T[:, np.newaxis, :]
        sigma[k] = np.sum(burden_rate[:, k, np.newaxis, np.newaxis] * post, axis=0) / s[k] + 0.001 * np.identity(size)

    # shape: 10 * 784 * 784
    return pi, mu, sigma


def gmm(data, seed, save_dir, k):
    centers = k_means(data, seed=seed, save_dir=save_dir, k=k)
    burden_rate = mask_one_hot(data, centers, k=k)
    print("B_shape0", burden_rate.shape)
    pi, mu, sigma = m_step(data.T, burden_rate, k)

    likelihood = log_likelihood(data, pi, mu, sigma, k)

    step = 0
    while step < 80:
        burden_rate = e_step(data.T, pi, mu, sigma, k)
        print("B_shape1", burden_rate.shape)
        pi, mu, sigma = m_step(data.T, burden_rate, k)

        print("step: {}, {}".format(step, likelihood))

        new_likelihood = log_likelihood(data, pi, mu, sigma, k)
        if likelihood - new_likelihood < 0.0001:
            print("======Clustering is converged=========")
            break
        likelihood = new_likelihood
        step += 1

    return burden_rate


def log_likelihood(data, pi, mu, sigma, K):
    s = .0
    print("mu: ", mu)
    for i in range(len(data)):
        tmp = .0
        for k in range(K):
            tmp += pi[k] * gaussian(data[i], mu[:, k], sigma[k])
        s += np.log2(tmp)
    return -s


if __name__ == "__main__":
    mnist = False

    if mnist:
        _data = load()
        train_data = _data["train_images"]/255.
        train_labels = _data["train_labels"]
        _k = 10
    else:
        _data = load_iris()
        train_data = _data["data"]
        train_labels = _data["target"]
        _k = 3

    _scores_df = pd.DataFrame()
    for _seed in range(100):
        print("seed: ", _seed)
        _file_path = os.path.dirname(os.path.abspath(__file__))
        if mnist:
            _save_dir = _file_path + "/data/seed_" + str(_seed)
        else:
            _save_dir = _file_path + "/data_iris/seed_" + str(_seed)

        _burden_rate = gmm(train_data, seed=_seed, save_dir=_save_dir, k=_k)

        _mask = np.argmax(_burden_rate, axis=1)
        _masks = np.array([_mask == mi for mi in range(_k)])

        draw_heatmap(train_labels, _masks, seed=_seed, save_dir=_save_dir, k=_k)
        _score = entropy_score(train_labels, _masks, k=_k)
        _score_df = pd.DataFrame({
            'seed': [_seed],
            'score': [_score]
        })
        _scores_df = _scores_df.append(_score_df)
        if mnist:
            _scores_df.to_csv(_file_path + "/data/gmm_score.csv")
        else:
            _scores_df.to_csv(_file_path + "/data_iris/gmm_score_iris.csv")
