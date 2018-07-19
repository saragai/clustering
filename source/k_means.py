# coding=UTF-8
import os
import os.path
import numpy as np
from PIL import Image
from load.load_mnist import load


def e_step(data, centers):
    # data[masks[arg_data]]
    distances = np.array([np.linalg.norm(data - center, axis=1) for center in centers]).transpose()
    mask = np.argmin(distances, axis=1)
    masks = np.array([mask == mi for mi in range(len(centers))])
    return masks


def m_step(data, masks):
    centers = np.array([np.mean(data[mask], axis=0) for mask in masks])
    return centers


def accuracy(labels, masks):
    print(np.array([[np.sum(labels[mask] == label) for label in range(10)] for mask in masks]))


def entropy(array):
    if array.ndim == 1:
        array.shape = (1, array.size)
    p = array/array.sum(axis=1).reshape(len(array), 1)
    log_p = np.log2(p)
    log_p[log_p == -np.inf] = 0
    return -np.sum(p * log_p, axis=1)


def entropy_score(result):
    weight = result.sum(axis=1)/result.sum()
    return np.sum(weight * entropy(result))


def draw_heat_map(result):
    pass


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    save_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/seed_" + str(seed)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    _data = load()
    train_images = _data["train_images"]/255.
    train_labels = _data["train_labels"]

    # _centers = np.random.rand(10, 784)
    _center_idx = [np.random.randint(0, len(train_images))]

    for _ in range(1, 10):
        _sqr_distance = np.array([np.sum(((train_images - train_images[idx]) ** 2), axis=1) for idx in _center_idx])
        nearest_sqr_distances = np.min(_sqr_distance.transpose(), axis=1)
        sum_sqr_distances = np.sum(nearest_sqr_distances)
        print(nearest_sqr_distances.shape)
        print(sum_sqr_distances.shape)
        _center_idx.append(np.random.choice(len(train_images), size=1, p=nearest_sqr_distances/sum_sqr_distances))

    _center_idx = np.array(_center_idx)
    print("center_idx:\n", _center_idx)
    print("center_label\n", train_labels[_center_idx])
    _centers = train_images[_center_idx]

    step = 0
    while step < 100:
        print("step: {}".format(step))
        _old_centers = _centers
        _masks = e_step(train_images, _centers)
        _centers = m_step(train_images, _masks)
        done = np.allclose(_old_centers, _centers)
        if step % 10 == 0 or done:
            print("Each cluster:")
            accuracy(train_labels, _masks)
            for i, _center in enumerate(_centers):
                image = np.uint8(_center.reshape((28, 28)) * 255)
                Image.fromarray(image).save(save_dir + '/img_{}_{}.jpg'.format(step, i))
        if done:
            print("Clustering converged")
            break
        step += 1
