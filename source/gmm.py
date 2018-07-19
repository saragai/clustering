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


def n(x, mu, sigma):
    d = len(mu)
    coefficient = 1/(2. * np.pi) ** (d * .5) / np.diag(sigma) ** .5
    return coefficient * np.exp(-.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu))


def e_step(data, pi, mu, sigma):
    burden_rate = np.array([[pi[k] * n(x[:, np.newaxis], mu[k], sigma[k]) for k in range(10)] for x in data.T])

    # N = n(data, mu, sigma)  # 60000 * 60000
    # burden_rate = pi * N / np.sum(pi * N)  # 10 * 60000

    return burden_rate


def m_step(data, burden_rate):
    s = np.sum(burden_rate, axis=0)
    pi = 1/len(data) * s  # shape: 10
    mu = data @ burden_rate / s  # shape: 784 * 10

    sigma = np.array([np.array([burden_rate[i, k] * (x - mu[:, k])[:, np.newaxis] @ (x - mu[:, k])[np.newaxis, :]
                                for i, x in enumerate(data.T)]).sum(axis=0)
                      / s[k] for k in range(10)])
    # shape: 10 * 784 * 784
    return pi, mu, sigma


def gmm(data, seed, save_dir):
    centers = k_means(data, seed=seed, save_dir=save_dir)
    burden_rate = mask_one_hot(data, centers)

    pi, mu, sigma = m_step(data.T, burden_rate)
    burden_rate = e_step(data.T, pi, mu, sigma)

    return burden_rate


if __name__ == "__main__":
    _data = load()
    train_images = _data["train_images"]/255.
    train_labels = _data["train_labels"]

    _scores_df = pd.DataFrame()
    for _seed in range(10):
        print("seed: ", _seed)
        _file_path = os.path.dirname(os.path.abspath(__file__))
        _save_dir = _file_path + "/data/seed_" + str(_seed)

        _centers = gmm(train_images[:200], seed=_seed, save_dir=_save_dir)
        draw_heatmap(train_images, train_labels, _centers, seed=_seed, save_dir=_save_dir)
        _score = entropy_score(train_images, train_labels, _centers)
        _score_df = pd.DataFrame({
            'seed': [_seed],
            'score': [_score]
        })
        _scores_df = _scores_df.append(_score_df)
        _scores_df.to_csv(_file_path + "/data/gmm_score.csv")
