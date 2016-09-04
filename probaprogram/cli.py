import matplotlib
matplotlib.use('agg')  # NOQA

from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam, RMSprop # NOQA
from helpers import data_discretization, DocumentVectorizer, generate_text, dispims_color, categ, generate_text_deterministic, dispims_color

import time
import numpy as np

from skimage.io import imsave

from shape import Sampler, Point, to_img, to_img2, render

class Vectorize(object):

    def __init__(self, max_nbparts, max_nbpoints):
        self.max_nbparts = max_nbparts
        self.max_nbpoints = max_nbpoints

    def fit(self, data):
        return self

    def transform(self, data):
        x = np.zeros((len(data), self.max_nbparts * self.max_nbpoints + self.max_nbparts, 3))
        x[:, :, 2] = 1
        for i, example in enumerate(data):
            k = 0
            for part in example:
                for point in part:
                    x[i, k, 0] = point.x
                    x[i, k, 1] = point.y
                    x[i, k, 2] = 0
                    k += 1
                x[i, k, 0] = 0
                x[i, k, 1] = 0
                x[i, k, 2] = 1
                k += 1
            x[i, k, 0] = 0
            x[i, k, 1] = 0
            x[i, k, 2] = 2
            k += 1
        return x

    def inverse_transform(self, X):
        data = []
        for x in X:
            parts = []
            part = []
            for cell in x:
                if cell[2] == 0:
                    x, y = cell[0], cell[1]
                    point = Point(x=x, y=y)
                    part.append(point)
                elif cell[2] == 1:
                    if len(part):
                        parts.append(part)
                    part = []
                else:
                    break
            data.append(parts)
        return data

class Discretize(object):

    def __init__(self, nbins=2, minval=0, maxval=1):
        self.bins = np.linspace(minval, maxval, nbins)

    def fit(self, data):
        return self

    def _transform(self, data, fn=lambda x:x):
        new_data = []
        for example in data:
            new_example = []
            for part in example:
                new_part = []
                for point in part:
                    x = fn(point.x)
                    y = fn(point.y)
                    new_point = Point(x=x, y=y)
                    new_part.append(new_point)
                new_example.append(new_part)
            new_data.append(new_example)
        return new_data

    def transform(self, data):
        def fn(x):
            return np.argmin(np.abs(self.bins - x))
        return self._transform(data, fn=fn)

    def inverse_transform(self, data):
        def fn(x):
            return self.bins[x]
        return self._transform(data, fn=fn)

def parts_key(parts):
    return (parts[0].x, parts[0].y)

def train():
    outdir = 'out'
    # Load data
    sampler = Sampler(nbparts=(1, 2), nbpoints=(1, 3))
    data = [sampler.sample() for i in range(10)]
    D = 20
    discretize = Discretize(nbins=D)
    vectorize = Vectorize(max_nbparts=sampler.nbparts[1], max_nbpoints=sampler.nbpoints[1])

    data = discretize.transform(data)
    data = vectorize.transform(data)
    # Params

    nb_epochs = 100000
    outdir = 'out'
    nb_hidden = 128
    batch_size = 128
    width, height = 32, 32
    thickness = 1
    T = data.shape[1]

    # Image model
    xim = Input(batch_shape=(batch_size, 1, width, height), dtype='float32')

    x = xim

    """
    x = Convolution2D(64, 5, 5)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Convolution2D(64, 5, 5)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Convolution2D(128, 5, 5)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    """
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    # Sequence Model
    x = RepeatVector(T)(x)
    h = x
    h = LSTM(100, init='orthogonal',return_sequences=True)(h)
    pred_x = Activation('softmax', name='x')(TimeDistributed(Dense(D))(h))
    pred_y = Activation('softmax', name='y')(TimeDistributed(Dense(D))(h))
    pred_stop = Activation('softmax', name='stop')(TimeDistributed(Dense(3))(h))

    h = LSTM(100, init='orthogonal')(h)
    h = Dense(1024)(h)
    h =  Activation('relu')(h)
    h = Dense(256)(h)
    h =  Activation('relu')(h)
    h = Dense(width * height)(h)
    h = Activation('sigmoid')(h)
    xrec = Reshape((1, width, height), name='xrec')(h)

    model = Model(input=xim, output=[pred_x, pred_y, pred_stop, xrec])


    optimizer = Adam(lr=0.001)
                    # rho=0.95,
                    # epsilon=1e-8)
    model.compile(
        loss={
            'x': 'categorical_crossentropy',
            'y': 'categorical_crossentropy',
            'stop': 'categorical_crossentropy',
            'xrec': 'mean_squared_error'
        },
        optimizer=optimizer)
    avg_loss = 0
    for epoch in range(nb_epochs):
        t = time.time()
        #sampler.rng = np.random.RandomState(42)
        data = [sorted(sampler.sample(), key=parts_key) for i in range(1024)]
        images = np.array([to_img2(render(d), w=width, h=height, thickness=thickness) for d in data])
        images = np.float32(images)
        images = images[:, np.newaxis, :, :]
        data_orig = data
        data = discretize.transform(data)
        data = vectorize.transform(data)
        data = np.int32(data)

        batch_losses = []
        for s in iterate_minibatches(len(images), batchsize=batch_size, exact=True):
            x_mb = images[s]
            y_mb = data[s]
            inputs = x_mb
            outputs = [categ(y_mb[:, :, 0], D=D), categ(y_mb[:, :, 1], D=D), categ(y_mb[:, :, 2], D=3), x_mb]
            model.fit(inputs, outputs, nb_epoch=1, batch_size=batch_size, verbose=0)
            loss = np.mean(model.evaluate(inputs, outputs, verbose=0, batch_size=batch_size))
            avg_loss = avg_loss * 0.999 + loss * 0.001
            batch_losses.append(loss)
        print('Mean loss : {}'.format(np.mean(batch_losses)))

        x_pred, y_pred, stop_pred, xrec = model.predict(images[0:128])

        x_pred = x_pred.argmax(axis=-1)[:, :, np.newaxis]
        y_pred = y_pred.argmax(axis=-1)[:, :, np.newaxis]
        stop_pred = stop_pred.argmax(axis=-1)[:, :, np.newaxis]
        pred = np.concatenate((x_pred, y_pred, stop_pred), axis=2)
        pred = vectorize.inverse_transform(pred)
        pred = discretize.inverse_transform(pred)
        pred_images = np.array([to_img2(render(d), w=width, h=height, thickness=thickness) for d in pred])
        real_images = images[0:128, 0]

        pred = pred_images[:, :, :, np.newaxis] * np.ones((1, 1, 1, 3))
        pred = dispims_color(pred, border=1, bordercolor=(10, 10, 10))

        real = real_images[:, :, :, np.newaxis] * np.ones((1, 1, 1, 3))
        real = dispims_color(real, border=1, bordercolor=(10, 10, 10))

        img = np.concatenate((pred, real), axis=1)
        imsave('{}/{:05d}.png'.format(outdir, epoch), img)


def iterate_minibatches(nb,  batchsize, exact=False):
    if exact:
        r = range(0, (nb/batchsize) * batchsize, batchsize)
    else:
        r = range(0, nb, batchsize)
    for start_idx in r:
        S = slice(start_idx, start_idx + batchsize)
        yield S

def floatX(x):
    return np.array(x).astype(np.float32)


if __name__ == '__main__':
    train()
