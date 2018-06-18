from keras import layers as kl
from keras.models import Model


def conventional():
    img_input = kl.Input((28, 28, 1))
    x = kl.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    x = kl.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = kl.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = kl.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = kl.GlobalAveragePooling2D()(x)

    x = kl.Dense(10, activation='softmax')(x)
    return Model(img_input, x)


def bias():
    img_input = kl.Input((28, 28, 1))
    x = kl.Conv2D(32, 3, padding='same')(img_input)
    x = kl.Conv2D(32, 3, padding='same')(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(64, 3, padding='same')(x)
    x = kl.Conv2D(64, 3, padding='same')(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(128, 3, padding='same')(x)
    x = kl.Conv2D(128, 3, padding='same')(x)
    x = kl.GlobalAveragePooling2D()(x)

    x = kl.Dense(10, activation='softmax')(x)
    return Model(img_input, x)


def bn():
    img_input = kl.Input((28, 28, 1))
    x = kl.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    x = kl.BatchNormalization()(x)
    x = kl.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.GlobalAveragePooling2D()(x)

    x = kl.Dense(10, activation='softmax')(x)
    return Model(img_input, x)


def skip():
    img_input = kl.Input((28, 28, 1))
    x = kl.Conv2D(32, 3, padding='same', use_bias=False)(img_input)
    x = kl.BatchNormalization()(x)
    x = kl.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    res = kl.Conv2D(64, 1, strides=2, use_bias=False)(x)
    res = kl.BatchNormalization()(res)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.add([x, res])
    res = kl.Conv2D(128, 1, strides=2, use_bias=False)(x)
    res = kl.BatchNormalization()(res)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = kl.add([x, res])
    x = kl.GlobalAveragePooling2D()(x)

    x = kl.Dense(10, activation='softmax')(x)
    return Model(img_input, x)
