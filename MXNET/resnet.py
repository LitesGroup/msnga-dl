import mxnet as mx
import cv2
import numpy as np
from collections import namedtuple
Batch = namedtuple('Batch',['data'])

def downloadModelParams():
    # Download model definition JSON.
    mx.test_utils.download('http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json')

    # Download pre optimised hyperparameters.
    mx.test_utils.download('http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params')

    # Download class labels for multi-class classifications.
    mx.test_utils.download('http://data.mxnet.io/models/imagenet-11k/synset.txt')
    pass


def loadModel():
    # Load model into memory and extract symbols, argument/auxiliary parameters
    symbols, argument_parameters, auxiliary_parameters = mx.model.load_checkpoint('resnet-152', 0)
    # Create an instance of the model.
    # If you are rich like @shoab10 use context=mx.gpu() because you own a GTX 1080Ti
    model = mx.mod.Module(symbol=symbols, context= mx.cpu(), label_names=None)
    # provide data and labels shapes to the model
    model.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=model._label_shapes)
    # set previously extracted parameters to avoid overfitting
    model.set_params(argument_parameters, auxiliary_parameters, allow_missing=True)
    return model


def loadLabels():
    # load previously downloaded labels
    with open('synset.txt', 'r') as f:
        labels = [a.rstrip() for a in f]
    return labels

def process_image(url, show=False):
    # Download the file to the local machine
    filename = mx.test_utils.download(url)
    # load image into opencv framework
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    # Dimentionality Reduction to feed into the model
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def make_prediction(url, model, labels):
    img = process_image(url, show=True)
    model.forward(Batch([mx.nd.array(img)]))
    prob = model.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))
    return a[0:5]

if __name__ == '__main__':
    downloadModelParams()
    mdl = loadModel()
    lbl = loadLabels()
    make_prediction("https://images.dailykos.com/images/178832/story_image/grumpycat.jpg", mdl, lbl)
