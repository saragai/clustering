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
from sklearn.datasets import load_iris


def mask_one_hot(data, centers, k=3):
    distances = np.array([np.linalg.norm(data - center, axis=1) for center in centers]).transpose()
    mask = np.argmin(distances, axis=1)
    masks = np.zeros([len(data), k])
    for i in range(len(data)):
        masks[i, mask[i]] = 1
    return masks


def e_step(data, centers):
    # data[masks[arg_data]]
    distances = np.array([np.linalg.norm(data - center, axis=1) for center in centers]).transpose()
    mask = np.argmin(distances, axis=1)
    masks = np.array([mask == mi for mi in range(len(centers))])
    return masks


def m_step(data, masks):
    centers = np.array([np.mean(data[mask], axis=0) for mask in masks])
    return centers


def entropy(array):
    if array.ndim == 1:
        array.shape = (1, array.size)
    p = array/array.sum(axis=1).reshape(len(array), 1)
    log_p = np.log2(p)
    log_p[log_p == -np.inf] = 0
    return -np.sum(p * log_p, axis=1)


def entropy_score(labels, masks, k=3):
    result = np.array([
        [np.sum(labels[mask] == label)/np.sum(labels == label) for label in range(k)]
        for mask in masks])
    weight = result.sum(axis=1)/result.sum()

    return np.sum(weight * entropy(result))


def draw_heatmap(labels, masks, save_dir="", seed=1, k=3, filename="/heatmap.png"):
    print(masks)
    raw_result = np.array([
        [np.sum(labels[mask] == label)/np.sum(labels == label) for label in range(k)]
        for mask in masks])

    # sort for figure
    result = np.zeros(raw_result.shape)
    for label in range(k):
        idx = np.argmax(raw_result[:, label])
        result[label] = raw_result[idx]
        raw_result[idx] = 0

    fig, ax = plt.subplots()

    sns.heatmap(result, ax=ax, cmap=plt.cm.Blues)

    ax.set_title("GMM heat map (seed={})".format(seed))
    ax.set_xlabel("label")
    ax.set_ylabel("cluster")

    ax.set_xticks(np.arange(result.shape[0]) + .5, minor=False)
    ax.set_yticks(np.arange(result.shape[1]) + .5, minor=False)

    # ax.invert_yaxis()
    # ax.xaxis.tick_top()

    ax.set_xticklabels(range(k), minor=False)
    ax.set_yticklabels([""]*k, minor=False)

    plt.savefig(save_dir + filename)


def k_means_pp_init(data, k):
    center_idx = [np.random.randint(0, len(data))]

    for _ in range(1, k):
        _sqr_distance = np.array([np.sum(((data - data[idx]) ** 2), axis=1) for idx in center_idx])
        nearest_sqr_distances = np.min(_sqr_distance.transpose(), axis=1)
        sum_sqr_distances = np.sum(nearest_sqr_distances)
        center_idx.append(np.random.choice(len(data), size=1, p=nearest_sqr_distances/sum_sqr_distances))

    center_idx = np.array(center_idx)
    centers = data[center_idx]
    return centers


def k_means(data, seed=1, save_dir="", k=3, is_center=False):
    np.random.seed(seed)
    save_name = save_dir + "/center.pkl"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(save_name):
        with open(save_name, 'rb') as f:
            centers = pickle.load(f)
            masks = e_step(data, centers)
        if is_center:
            return centers
        return masks

    # centers = np.random.rand(10, 784)
    centers = k_means_pp_init(data, k=k)
    masks = e_step(data, centers)

    step = 0
    while step < 80:
        print("step: {}".format(step))
        old_centers = centers
        masks = e_step(data, centers)
        centers = m_step(data, masks)
        done = np.allclose(old_centers, centers)
        if step % 10 == 0 or done:
            for i, center in enumerate(centers):
                if mnist:
                    image = np.uint8(center.reshape((28, 28)) * 255)
                    Image.fromarray(image).save(save_dir + '/img_{}_{}.jpg'.format(step, i))
        if done:
            print("Clustering is converged")
            break
        step += 1

    with open(save_name, 'wb') as f:
        print("make pickle file")
        pickle.dump(centers, f, -1)

    return masks


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

        _masks = k_means(train_data, seed=_seed, save_dir=_save_dir, k=_k)

        draw_heatmap(train_labels, _masks, seed=_seed, save_dir=_save_dir, k=_k)
        _score = entropy_score(train_labels, _masks, k=_k)
        _score_df = pd.DataFrame({
            'seed': [_seed],
            'score': [_score]
        })
        _scores_df = _scores_df.append(_score_df)
        if mnist:
            _scores_df.to_csv(_file_path + "/data/score.csv")
        else:
            _scores_df.to_csv(_file_path + "/data_iris/score_iris.csv")
