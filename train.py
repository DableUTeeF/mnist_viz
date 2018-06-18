import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
from models import *
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

modelss = [skip]
names = ['skip']

if __name__ == '__main__':
    for i in range(len(modelss)):
        model = modelss[i]()
        model.compile(optimizer=SGD(lr=0.01, decay=1e-5, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
        x_train = x_train.astype('float32')
        x_train /= 255.
        x_train -= 0.5
        x_train *= 2
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
        x_test = x_test.astype('float32')
        x_test /= 255
        x_test -= 0.5
        x_test *= 2
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        log = model.fit(x=x_train,
                        y=y_train,
                        batch_size=32,
                        epochs=5,
                        verbose=1,
                        validation_data=(x_test, y_test))
        model.save_weights('{}.h5'.format(names[i]))
        with open('{}.txt'.format(names[i]), 'w') as wr:
            wr.write(str(log))
