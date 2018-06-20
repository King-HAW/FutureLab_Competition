from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Average
from skimage import io, transform

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

w = 299
h = 299
class_num = 20

# Hyperparameter
batch_size = 5


def read_img(list_path, data_path):
    list_data = pd.read_csv(list_path, header=0)
    list_data = np.array(list_data, dtype=str)
    imgs = []
    data_name_c = []
    data_name_g = []
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
            data_name_g.append(list_data[i, 0])
        else:
            imgs.append(img)
            data_name_c.append(list_data[i, 0])
    name = ['FILE_ID']
    tmp1 = pd.DataFrame(columns=name, data=data_name_c)
    tmp1.to_csv('list_nogrey_testb.csv', index=None)
    tmp2 = pd.DataFrame(columns=name, data=data_name_g)
    tmp2.to_csv('list_grey_testb.csv', index=None)
    return np.asarray(imgs, np.float32)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def generate_batch_data(x, batch_size):
    ylen = x.shape[0]
    loopcount = ylen // batch_size
    while 1:
        for i in range(0, loopcount):
            yield x[i * batch_size:(i + 1) * batch_size]


def ensemble_model(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)

    model = Model(model_input, y, name='ensemble')

    return model


def inceptionv3_ft(model_input):
    base_model = InceptionV3(input_tensor=model_input, weights=None, include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(class_num, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=predictions)

    return model


def inception_resnet_v2_ft(model_input):
    base_model = InceptionResNetV2(input_tensor=model_input, weights=None, include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(class_num, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=predictions)

    return model


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        help="Path to directory of training set or test set, depends on the running mode.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_prediction/', help="Path to directory of checkpoint.")
    parser.add_argument('--target_file', type=str, default='./testb_predict_results.csv', help='Path to test result file.')

    args = parser.parse_args()

    list_path = args.dataset_dir + 'list.csv'
    data_path = args.dataset_dir + 'data/'
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model_input = Input(shape=(w, h, 3))

    model_1 = inceptionv3_ft(model_input)
    model_2 = inception_resnet_v2_ft(model_input)

    model_1.load_weights(args.checkpoint_dir + 'weights-best-inception-v3-ft-futurelab-150.hdf5')
    model_2.load_weights(args.checkpoint_dir + 'weights-best-inception-resnet-v2-ft-futurelab.hdf5')

    models = [model_1, model_2]
    ensemble_model = ensemble_model(models, model_input)

    data = read_img(list_path, data_path)
    data = preprocess_input(data)

    ypredraw = ensemble_model.predict_generator(generator=generate_batch_data(data, batch_size=batch_size),
                                                steps=data.shape[0] // batch_size)
    # np.save("ypredraw.npy", ypredraw)

    print("Compelete!")
    ypred_temp = np.argsort(ypredraw, axis=1)
    ypred1 = np.expand_dims(ypred_temp[:,-1], axis=1)
    ypred2 = np.expand_dims(ypred_temp[:,-2], axis=1)
    ypred3 = np.expand_dims(ypred_temp[:,-3], axis=1)
    ypred_top3 = np.concatenate((ypred1, ypred2, ypred3), axis=1)
    # np.save("ypred_top3.npy", ypred_top3)

    list_name = pd.read_csv(list_path, header=0)
    list_name = np.array(list_name, dtype=str)
    result_title = ['FILE_ID', 'CATEGORY_ID0', 'CATEGORY_ID1', 'CATEGORY_ID2']
    result = np.concatenate((np.expand_dims(list_name[:, 0], axis=1), ypred_top3), axis=1)
    result_final = pd.DataFrame(columns=result_title, data=result)
    result_final.to_csv(args.target_file, index=None)
