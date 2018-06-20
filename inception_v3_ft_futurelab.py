from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import SGD
from skimage import io, transform
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import functools
import argparse

w = 299
h = 299
class_num = 20

# Hyperparameter
epochs = 150
batch_size = 32


def read_img(list_path, data_path):
    list_data = pd.read_csv(list_path, header=0)
    list_data = np.array(list_data, dtype=str)
    imgs = []
    labels = []
    for i in range(0, list_data.shape[0]):
        print('reading the images:%s' % (list_data[i, 0] + '.jpg_{}'.format(i)))
        img = io.imread(data_path + list_data[i, 0] + '.jpg')
        img = transform.resize(img, (w, h))
        print(img.shape)
        if len(img.shape) != 3:
            img_tmp = np.zeros((w, h, 3))
            img_tmp[:, :, 0] = img
            img_tmp[:, :, 1] = img
            img_tmp[:, :, 2] = img
            imgs.append(img_tmp)
            labels.append(int(list_data[i, 1]))
        else:
            imgs.append(img)
            labels.append(int(list_data[i, 1]))
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def shuffle_data(data, label):
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    data_tmp = data[index]
    label_tmp = label[index]
    return data_tmp, label_tmp


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        help="Path to directory of training set or test set, depends on the running mode.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help="Path to directory of checkpoint.")

    args = parser.parse_args()

    list_path = args.dataset_dir + 'list.csv'
    data_path = args.dataset_dir + 'data/'

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # model define
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(class_num, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    # load data and preprocessing
    data, label = read_img(list_path, data_path)
    data = preprocess_input(data)
    label = np.expand_dims(label, axis=1)
    data, label = shuffle_data(data, label)
    label = to_categorical(label, num_classes=class_num)

    # cut data into train set and validation set
    x_tr, x_val, y_tr, y_val = train_test_split(data, label, test_size=0.1, random_state=111)

    # fine tuning
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
    top3_acc.__name__ = 'top3_acc'

    model.compile(optimizer=SGD(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy', top3_acc])

    callbacks = [ModelCheckpoint(args.checkpoint_dir + 'weights-best-inception-v3-ft-futurelab-150.hdf5',
                                 monitor='val_acc', verbose=1)]

    # data augmentation
    train_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_datagen.fit(x_tr)

    history = model.fit_generator(generator=train_datagen.flow(x_tr, y_tr, batch_size=batch_size),
                                  steps_per_epoch=(x_tr.shape[0] // batch_size)*2,
                                  callbacks=callbacks, validation_data=(x_val, y_val), epochs=epochs, verbose=2)

    train_acc = np.array(history.history['acc'])
    train_loss = np.array(history.history['loss'])
    np.savetxt('train-inception-v3.txt', (train_acc, train_loss))
    val_acc = np.array(history.history['val_acc'])
    val_loss = np.array(history.history['val_loss'])
    np.savetxt('val-inception-v3.txt', (val_acc, val_loss))
