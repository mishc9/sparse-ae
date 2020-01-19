from typing import Union, Mapping, Sized

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.datasets import mnist


def load_mnist(shape):
    """
    Загрузка и предобработка датасета MNIST
    :param shape: целевая размерность датасета
    :return: подготовленный датасет
    """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255.
    X_test = X_test / 255.
    X_train = X_train.reshape(-1, shape)
    X_test = X_test.reshape(-1, shape)
    return (X_train, Y_train), (X_test, Y_test)


def get_sample(values, sample_size):
    """
    Извлекает заданное количество случайных сэмплов из дадасета случайным образом
    :param values: множество значений (данные)
    :param sample_size: количество значений в сэмпле
    :return: выбранное подмножетсво значений
    """
    return np.random.choice(values, sample_size)


def get_one_item_of_each_class(values, labels):
    """
    Извлекает по одному случайному изображению для каждого из классов MNIST, возвращает в
    отсортированном порядке.
    :param values: множество значений (данные)
    :param labels: метки для данных
    :return: отсортированный по меткам набор изображений
    """
    output = []
    label_set = []
    idx = np.arange(len(values))
    np.random.shuffle(idx)
    values = values[idx]
    labels = labels[idx]
    for (val, label) in zip(values, labels):
        if label not in label_set:
            label_set.append(label)
            output.append(val)
            if len(label_set) == 10:
                break

    sorted_labels = np.argsort(label_set)
    return np.array(output)[sorted_labels]


def train_val_split(x, train_size: float):
    """
    Разделяет выборку на обучающую и валидационную.
    :param x: выборка для разделения
    :param train_size: float в диапазоне [0, 1] - размер обучающей выборки
    :return: пара (обучающая выборка, валидационная выборка)
    """
    length = len(x)
    idx = np.arange(length)
    np.random.shuffle(idx)
    train_idx = idx[:int(length * train_size)]
    val_idx = idx[int(length * train_size):]
    train = x[train_idx]
    val = x[val_idx]
    return train, val


def reconstruct_from_top_k(encoder: Model, decoder: Model, k: int, images: np.array) -> np.array:
    """
    Реконструирует изображения из k наибольших активаций в латентном слое.
    :param encoder: обученный энкодер
    :param decoder: обученный декодер
    :param k: количество нейронов, котоыре не будут обнулены
    :param images: изображения для реконструкции
    :return: реконструированные изображения
    """
    latent = encoder.predict(images)
    for i in range(len(images)):
        argsort = np.argsort(latent[i])
        nullify = argsort[:-k]
        latent[i, nullify] = 0
    reconstructed = decoder.predict(latent)
    return reconstructed


def plot_reconstructed_from_top_k(images_list: Union[Mapping, Sized], save_to: str, image_shape, n_active: int):
    """
    Реконструирует изображение, обнуляя k наименьших активаций в латентном слое
    :param images_list: набор изображений для ркконструкции
    :param save_to: путь для сохранения файла
    :param image_shape: размерность изображения
    :param n_active: количество активных нейронов
    :return: None
    """
    _, images = next(iter(images_list.items()))
    f, axes = plt.subplots(len(images_list), len(images), sharey=True)
    image_shape = [-1] + list(image_shape)

    for idx, (path_to_model, images) in enumerate(images_list.items()):
        images = images.reshape(*image_shape)
        for j, img in enumerate(images):
            axes[idx, j].matshow(img, cmap='gray')
            axes[idx, j].set_xticks([])
            axes[idx, j].set_yticks([])

            if j == 0:
                axes[idx, j].set_ylabel(f'{path_to_model.split("/")[-1]}',
                                        rotation=90,
                                        size=8)

    f.suptitle(f'Reconstructed from top {n_active} neurons from hidden layer')
    f.savefig(save_to)
    plt.close(f)


def plot_latent_unit_activation(decoder: tf.keras.Model, latent_codes, save_to, image_shape):
    """
    Реконструирует и созраняет изображение по активациям латентного слоя
    :param decoder: обученный декодер
    :param latent_codes: батч латентных представлений (возможно, сгенерированный)
    :param save_to: путь изображения
    :param image_shape: размерность реконструированного изображения
    :return: None
    """
    reconstructed = decoder.predict(latent_codes)
    f, axes = plt.subplots(8, 8, sharey=True)

    image_shape = [-1] + list(image_shape)
    reconstructed = reconstructed.reshape(*image_shape)

    for i in range(8):
        for j in range(8):
            axes[i, j].matshow(reconstructed[i * 8 + j], cmap='gray')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    f.savefig(save_to)
    plt.close(f)


def plot_reconstruction(autoencoder: tf.keras.Model, images, save_to, image_shape):
    """
    Реконструирует и созраняет изображение по активациям латентного слоя
    :param autoencoder: обученный автокодировщик
    :param images: батч изображений
    :param save_to: путь изображения
    :param image_shape: размерность реконструированного изображения
    :return: None
    """
    reconstructed = autoencoder.predict(images)
    f, axes = plt.subplots(2, len(images), sharey=True)

    image_shape = [-1] + list(image_shape)
    reconstructed = reconstructed.reshape(*image_shape)
    images = images.reshape(*image_shape)

    for idx, (img, rec) in enumerate(zip(images, reconstructed)):
        title_opts = dict(y=1.1, rotation=30, size=8)
        axes[0, idx].matshow(img)
        axes[0, idx].set_title(f"Original {idx}", **title_opts)
        axes[1, idx].matshow(rec)
        axes[1, idx].set_title(f"Reconstructed {idx}", **title_opts)
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])

    f.savefig(save_to)
    plt.close(f)
