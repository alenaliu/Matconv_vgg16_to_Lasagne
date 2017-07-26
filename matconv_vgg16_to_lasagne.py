import numpy as np
import scipy.io as sio
import theano
import skimage.transform as tns
import theano.tensor as T
import lasagne
import scipy.misc as misc
import matplotlib.pyplot as plt
import urllib
from lasagne.utils import floatX
import io


def prep_image(url,avg_im):
    # Read URL
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)

    raw_image = np.copy(misc.imresize(im, (224, 224, 3)))
    
    # Note that in matlab,bilinear interpolation (not default) must be used
    # for imresize
    im = im.astype(np.float32)
    im = tns.resize(im, (224, 224, 3), preserve_range=True)

    # Subtract the average image
    im[:, :, 0] = im[:, :, 0] - avg_im[0]
    im[:, :, 1] = im[:, :, 1] - avg_im[1]
    im[:, :, 2] = im[:, :, 2] - avg_im[2]

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    return raw_image, floatX(im[np.newaxis])


# Lasagne stores weights with number of filters as first dim
# Matconv stores number of filters as last dim- this converts dim order
# of Matconv weights to Lasagne format
def change_conv_weights(weights):
    num_filters = weights.shape[3]
    f_size = weights.shape[0]
    prev_f = weights.shape[2]
    new_weights = np.zeros((num_filters, prev_f, f_size, f_size)).astype(np.float32)
    
    for k in range(num_filters):
        single_filter = weights[:, :, :, k]
        single_filter = np.swapaxes(np.swapaxes(single_filter, 1, 2), 0, 1)
        new_weights[k, :, :, :] = single_filter
    return new_weights    
    

# Builds a dictionary of MatConvNet weights moved into Lasagne format
def build_weights(layers):
    
    params = {}
    conv1_1w = np.array(layers[0, 0]['weights'][0, 0])[0, 0]
    params['conv1_1w'] = change_conv_weights(conv1_1w)
    
    conv1_1b = np.array(layers[0, 0]['weights'][0, 0])[0, 1]
    params['conv1_1b'] = np.squeeze(conv1_1b)
    
    conv1_2w = np.array(layers[0, 2]['weights'][0, 0])[0, 0]
    params['conv1_2w'] = change_conv_weights(conv1_2w)
    
    conv1_2b = np.array(layers[0, 2]['weights'][0, 0])[0, 1]
    params['conv1_2b'] = np.squeeze(conv1_2b)

    conv2_1w = np.array(layers[0, 5]['weights'][0, 0])[0, 0]
    params['conv2_1w'] = change_conv_weights(conv2_1w)
    
    conv2_1b = np.array(layers[0, 5]['weights'][0, 0])[0, 1]
    params['conv2_1b'] = np.squeeze(conv2_1b) 

    conv2_2w = np.array(layers[0, 7]['weights'][0, 0])[0, 0]
    params['conv2_2w'] = change_conv_weights(conv2_2w)
    
    conv2_2b = np.array(layers[0, 7]['weights'][0, 0])[0, 1]
    params['conv2_2b'] = np.squeeze(conv2_2b)
    
    conv3_1w = np.array(layers[0, 10]['weights'][0, 0])[0, 0]
    params['conv3_1w'] = change_conv_weights(conv3_1w)
    
    conv3_1b = np.array(layers[0, 10]['weights'][0, 0])[0, 1]
    params['conv3_1b'] = np.squeeze(conv3_1b)
    
    conv3_2w = np.array(layers[0, 12]['weights'][0, 0])[0, 0]
    params['conv3_2w'] = change_conv_weights(conv3_2w)
    
    conv3_2b = np.array(layers[0, 12]['weights'][0, 0])[0, 1]
    params['conv3_2b'] = np.squeeze(conv3_2b)
    
    conv3_3w = np.array(layers[0, 14]['weights'][0, 0])[0, 0]
    params['conv3_3w'] = change_conv_weights(conv3_3w)
    
    conv3_3b = np.array(layers[0, 14]['weights'][0, 0])[0, 1]
    params['conv3_3b'] = np.squeeze(conv3_3b)
    
    conv4_1w = np.array(layers[0, 17]['weights'][0, 0])[0, 0]
    params['conv4_1w'] = change_conv_weights(conv4_1w)
    
    conv4_1b = np.array(layers[0, 17]['weights'][0, 0])[0, 1]
    params['conv4_1b'] = np.squeeze(conv4_1b)
    
    conv4_2w = np.array(layers[0, 19]['weights'][0, 0])[0, 0]
    params['conv4_2w'] = change_conv_weights(conv4_2w)
    
    conv4_2b = np.array(layers[0, 19]['weights'][0, 0])[0, 1]
    params['conv4_2b'] = np.squeeze(conv4_2b)
    
    conv4_3w = np.array(layers[0, 21]['weights'][0, 0])[0, 0]
    params['conv4_3w'] = change_conv_weights(conv4_3w)
    
    conv4_3b = np.array(layers[0, 21]['weights'][0, 0])[0, 1]
    params['conv4_3b'] = np.squeeze(conv4_3b)
    
    conv5_1w = np.array(layers[0, 24]['weights'][0, 0])[0, 0]
    params['conv5_1w'] = change_conv_weights(conv5_1w)
    
    conv5_1b = np.array(layers[0, 24]['weights'][0, 0])[0, 1]
    params['conv5_1b'] = np.squeeze(conv5_1b)
    
    conv5_2w = np.array(layers[0, 26]['weights'][0, 0])[0, 0]
    params['conv5_2w'] = change_conv_weights(conv5_2w)
    
    conv5_2b = np.array(layers[0, 26]['weights'][0, 0])[0, 1]
    params['conv5_2b'] = np.squeeze(conv5_2b)
    
    conv5_3w = np.array(layers[0, 28]['weights'][0, 0])[0, 0]
    params['conv5_3w'] = change_conv_weights(conv5_3w)
    
    conv5_3b = np.array(layers[0, 28]['weights'][0, 0])[0, 1]
    params['conv5_3b'] = np.squeeze(conv5_3b)
    
    fc6_w = np.array(layers[0, 31]['weights'][0, 0])[0, 0]
    params['fc6_w'] = change_conv_weights(fc6_w)
    
    fc6_b = np.array(layers[0, 31]['weights'][0, 0])[0, 1]
    params['fc6_b'] = np.squeeze(fc6_b)
    
    fc7_w = np.array(layers[0, 33]['weights'][0, 0])[0, 0]
    params['fc7_w'] = np.squeeze(fc7_w)
    
    fc7_b = np.array(layers[0, 33]['weights'][0, 0])[0, 1]
    params['fc7_b'] = np.squeeze(fc7_b)
    
    fc8_w = np.array(layers[0, 35]['weights'][0, 0])[0, 0]
    params['fc8_w'] = np.squeeze(fc8_w)
    
    fc8_b = np.array(layers[0, 35]['weights'][0, 0])[0, 1]
    params['fc8_b'] = np.squeeze(fc8_b)
    return params


# Defines network matching MatconvNet's VGG-16 format
def build_cnn(params):
    net = {}
    input_var = T.tensor4('inputs')
    net['input'] = lasagne.layers.InputLayer(shape=(None, 3, 224, 224),
    input_var=input_var)
    
    net['conv1_1'] = lasagne.layers.Conv2DLayer(net['input'], num_filters=64, filter_size=3, pad=1, stride=1,
                                                W=params['conv1_1w'], b=params['conv1_1b'], flip_filters=False)
    net['conv1_2'] = lasagne.layers.Conv2DLayer(net['conv1_1'], num_filters=64, filter_size=3, pad=1, stride=1,
                                                W=params['conv1_2w'], b=params['conv1_2b'], flip_filters=False)
    
    # Matconv uses [0 1 0 1] padding for maxpool    
    net['pd1'] = lasagne.layers.PadLayer(net['conv1_2'], width=[(0, 1), (0, 1)], batch_ndim=2)
    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['pd1'], pool_size=2, stride=2, pad=(0, 0), ignore_border=True)
    
    net['conv2_1'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=128, filter_size=3, pad=1, stride=1,
                                                W=params['conv2_1w'], b=params['conv2_1b'], flip_filters=False)
    net['conv2_2'] = lasagne.layers.Conv2DLayer(net['conv2_1'], num_filters=128, filter_size=3, pad=1, stride=1,
                                                W=params['conv2_2w'], b=params['conv2_2b'], flip_filters=False)
    net['pd2'] = lasagne.layers.PadLayer(net['conv2_2'], width=[(0, 1), (0, 1)], batch_ndim=2)
    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['pd2'], pool_size=2, stride=2, pad=(0, 0), ignore_border=True)

    net['conv3_1'] = lasagne.layers.Conv2DLayer(net['pool2'], num_filters=256, filter_size=3, pad=1, stride=1,
                                                W=params['conv3_1w'], b=params['conv3_1b'], flip_filters=False)
    net['conv3_2'] = lasagne.layers.Conv2DLayer(net['conv3_1'], num_filters=256, filter_size=3, pad=1, stride=1,
                                                W=params['conv3_2w'], b=params['conv3_2b'], flip_filters=False)
    net['conv3_3'] = lasagne.layers.Conv2DLayer(net['conv3_2'], num_filters=256, filter_size=3, pad=1, stride=1,
                                                W=params['conv3_3w'], b=params['conv3_3b'], flip_filters=False)
    net['pd3'] = lasagne.layers.PadLayer(net['conv3_3'], width=[(0, 1), (0, 1)], batch_ndim=2)
    net['pool3'] = lasagne.layers.MaxPool2DLayer(net['pd3'], pool_size=2, stride=2, pad=(0, 0), ignore_border=True)
    
    net['conv4_1'] = lasagne.layers.Conv2DLayer(net['pool3'], num_filters=512, filter_size=3, pad=1, stride=1,
                                                W=params['conv4_1w'], b=params['conv4_1b'], flip_filters=False)
    net['conv4_2'] = lasagne.layers.Conv2DLayer(net['conv4_1'], num_filters=512, filter_size=3, pad=1, stride=1,
                                                W=params['conv4_2w'], b=params['conv4_2b'], flip_filters=False)
    net['conv4_3'] = lasagne.layers.Conv2DLayer(net['conv4_2'], num_filters=512, filter_size=3, pad=1, stride=1,
                                                W=params['conv4_3w'], b=params['conv4_3b'], flip_filters=False)
    net['pd4'] = lasagne.layers.PadLayer(net['conv4_3'], width=[(0, 1), (0, 1)], batch_ndim=2)
    net['pool4'] = lasagne.layers.MaxPool2DLayer(net['pd4'], pool_size=2, stride=2, pad=(0, 0), ignore_border=True)
    
    net['conv5_1'] = lasagne.layers.Conv2DLayer(net['pool4'], num_filters=512, filter_size=3, pad=1, stride=1,
                                                W=params['conv5_1w'], b=params['conv5_1b'], flip_filters=False)
    net['conv5_2'] = lasagne.layers.Conv2DLayer(net['conv5_1'], num_filters=512, filter_size=3, pad=1, stride=1,
                                                W=params['conv5_2w'], b=params['conv5_2b'], flip_filters=False)
    net['conv5_3'] = lasagne.layers.Conv2DLayer(net['conv5_2'], num_filters=512, filter_size=3, pad=1, stride=1,
                                                W=params['conv5_3w'], b=params['conv5_3b'], flip_filters=False)
    net['pd5'] = lasagne.layers.PadLayer(net['conv5_3'], width=[(0, 1), (0, 1)], batch_ndim=2)
    net['pool5'] = lasagne.layers.MaxPool2DLayer(net['pd5'], pool_size=2, stride=2, pad=(0, 0), ignore_border=True)
    
    # First dense layer must be defined as a conv layer to match matconv format
    net['fc6'] = lasagne.layers.Conv2DLayer(net['pool5'], num_filters=4096, filter_size=7, pad=0,
                                            W=params['fc6_w'], b=params['fc6_b'], flip_filters=False)
    net['fc6_r'] = lasagne.layers.flatten(net['fc6'])    
    
    net['fc7'] = lasagne.layers.DenseLayer(net['fc6_r'], num_units=4096, W=params['fc7_w'], b=params['fc7_b'])   
    net['fc8'] = lasagne.layers.DenseLayer(net['fc7'], num_units=1000, W=params['fc8_w'], b=params['fc8_b'],
                                           nonlinearity=lasagne.nonlinearities.softmax)
    out = net['fc8']    
    
    return out


# We define 100 urls to check our model against
def get_random_urls():
    index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
    image_urls = index.split('<br>')
    np.random.seed(23)
    np.random.shuffle(image_urls)
    image_urls = image_urls[:100]
    sio.savemat('urls.mat', {'urls': image_urls})
    return image_urls


def main():
    try:
        mat_file = sio.loadmat('imagenet-vgg-verydeep-16.mat')
    except IOError:
        urllib.urlretrieve('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat',
                           'imagenet-vgg-verydeep-16.mat')
        mat_file = sio.loadmat('imagenet-vgg-verydeep-16.mat')  

    # Extracts the average image + param weights from matconvnet    
    layers = mat_file['layers']
    avg_im = np.squeeze(np.array(mat_file['meta']['normalization'][0, 0]['averageImage'][0, 0]).astype(np.float32))
    classes = mat_file['meta']['classes'][0, 0]['description'][0, 0].tolist()
    classes = classes[0]
    
    params = build_weights(layers)
    out = build_cnn(params)
    
    # Load 100 random URLs to check both models give similar results
    try:
        image_urls = sio.loadmat('urls.mat')['urls']
    except IOError:
        image_urls = get_random_urls()
    all_probs = np.zeros((1000, 0), dtype=np.float32)
    
    # All probabilities have been evaluated using matconvnet beta 15
    matconv_probs = np.asarray(sio.loadmat('matconv_probs.mat')['all_scores'], dtype=np.float32)
    for n in range(0, len(image_urls)):
        url = image_urls[n]
        try:
            raw_image, im = prep_image(url, avg_im)
    
            probs = np.array(lasagne.layers.get_output(out, im, deterministic=True).eval())
            probs = np.expand_dims(np.squeeze(probs), 1)
        
        except IOError:
            print('bad url: ' + url)
            probs = np.zeros((1000, 1), dtype=np.float32)
        
        all_probs = np.concatenate((all_probs, probs), 1)
        
        # Shows which images do not match between the two models
        # Skips images matlab was unable to load
        if not np.allclose(probs[:, 0], matconv_probs[:, n]) and np.sum(matconv_probs[:, n]) > 0:
            plt.figure()
            plt.imshow(raw_image)
            plt.text(0, 0, classes[np.argmax(probs)], color='red', bbox=dict(facecolor='white', alpha=1))
            plt.text(0, 224, classes[np.argmax(matconv_probs[:, n])], color='black', bbox=dict(facecolor='white',
                                                                                               alpha=1))
            plt.axis('off')
    
main()
    
    
