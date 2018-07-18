# coding=UTF-8
import numpy as np
from PIL import Image
from load.load_mnist import load


def e_step(data, centers):
    # data[masks[arg_data]]
    distances = np.array([np.linalg.norm(data - center, axis=1) for center in centers]).transpose()
    mask = np.argmin(distances, axis=1)
    masks = np.array([mask == mi for mi in range(len(centers))])
    print("masks:", masks.shape)
    return masks


def m_step(data, masks):
    centers = np.array([np.mean(data[mask], axis=0) for mask in masks])
    print("centers:", centers.shape)
    return centers


def accuracy(data, labels, centers):
    pass


if __name__ == "__main__":
    _data = load()
    train_images = _data["train_images"]/255
    train_labels = _data["train_labels"]

    np.random.seed(1)
    _centers = np.random.rand(10, 784)
    # _centers = [np.random.randint(0, len(train_images))]

    step = 0
    while step < 100:
        _old_centers = _centers
        _masks = e_step(train_images, _centers)
        _centers = m_step(train_images, _masks)
        if step % 10 == 0:
            for i, _center in enumerate(_centers):
                Image.fromarray(np.uint8(_center.reshape((28, 28)) * 255)).save('img_{}_{}.jpg'.format(step, i))
        step += 1
        if np.allclose(_old_centers, _centers):
            break
