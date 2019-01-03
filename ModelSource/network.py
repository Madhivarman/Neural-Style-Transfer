import scipy.io
import time
import tensorflow as tf 
import numpy as np

class Network():

    def __init__(self, parser):
        print('Building VGG-19 Network')
        self.parser = parser

    # <!--CONV LAYER-->

    def conv_layer(
        self,
        layer_name,
        layer_input,
        W,
        ):
        conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1],
                            padding='SAME')

        print('--{} | shape={} | weights_shape={}'.format(layer_name,
                conv.get_shape(), W.get_shape()))

        return conv

    # <!-- RELU LAYER-->

    def relu_layer(
        self,
        layer_name,
        layer_input,
        b,
        ):

        relu = tf.nn.relu(layer_input + b)

        print('--{} | shape={} | Bias_shape={}'.format(layer_name,
                relu.get_shape(), b.get_shape()))

        return relu

    # <!-- POOLING LAYER-->

    def pool_layer(
        self,
        name,
        layer_input,
        args,
        ):

        if args.poolingChoice == 'avg':
            pool = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')
        elif args.poolingChoice == 'max':

            pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        print('--{} | Shape={}'.format(name, layer_input.get_shape()))

        return pool

    # <!-- GET WEIGHTS-->

    def get_weights(self, vgg_layers, i):
        weights = vgg_layers[i][0][0][2][0][0]
        W = tf.constant(weights)

        return W

    # <!-- GET BIASES-->

    def get_bias(self, vgg_layers, i):
        bias = vgg_layers[i][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, bias.size))

        return b

    def buildModel(self, input_img):
        net = {}
        _, h, w, d = input_img.shape

        tick = time.time()

        print('--- LOADING THE VGG19 MODELS ----')

        vgg_rawnet = scipy.io.loadmat(self.parser.modelweights)

        tock = time.time()
        print('To Load Pre Train VGG19 model it took {} Seconds'.format(tock - tick))
        print('---- MODEL LOADED ----')

        vgg_layers = vgg_rawnet['layers'][0]
        net['input'] = tf.Variable(np.zeros((1, h, w, d),
                                   dtype=np.float32))  # assign with random zeros

        # layer 1

        net['conv1_1'] = self.conv_layer('conv1_1', net['input'],
                W=self.get_weights(vgg_layers, 0))
        net['relu1_1'] = self.relu_layer('relu_1', net['conv1_1'],
                b=self.get_bias(vgg_layers, 0))
        net['conv1_2'] = self.conv_layer('conv1_2', net['relu1_1'],
                W=self.get_weights(vgg_layers, 2))
        net['relu1_2'] = self.relu_layer('relu1_2', net['conv1_2'],
                b=self.get_bias(vgg_layers, 2))
        net['pool1'] = self.pool_layer('pool1', net['relu1_2'], self.parser)

          # layer2

        net['conv2_1'] = self.conv_layer('conv2_1', net['pool1'],
                W=self.get_weights(vgg_layers, 5))
        net['conv2_1'] = self.conv_layer('conv2_1', net['pool1'],
                W=self.get_weights(vgg_layers, 5))
        net['relu2_1'] = self.relu_layer('relu2_1', net['conv2_1'],
                b=self.get_bias(vgg_layers, 5))
        net['conv2_2'] = self.conv_layer('conv2_2', net['relu2_1'],
                W=self.get_weights(vgg_layers, 7))
        net['relu2_2'] = self.relu_layer('relu2_2', net['conv2_2'],
                b=self.get_bias(vgg_layers, 7))
        net['pool2'] = self.pool_layer('pool2', net['relu2_2'], self.parser)

          # layer3

        net['conv3_1'] = self.conv_layer('conv3_1', net['pool2'],
                W=self.get_weights(vgg_layers, 10))
        net['relu3_1'] = self.relu_layer('relu3_1', net['conv3_1'],
                b=self.get_bias(vgg_layers, 10))
        net['conv3_2'] = self.conv_layer('conv3_2', net['relu3_1'],
                W=self.get_weights(vgg_layers, 12))
        net['relu3_2'] = self.relu_layer('relu3_2', net['conv3_2'],
                b=self.get_bias(vgg_layers, 12))
        net['conv3_3'] = self.conv_layer('conv3_3', net['relu3_2'],
                W=self.get_weights(vgg_layers, 14))
        net['relu3_3'] = self.relu_layer('relu3_3', net['conv3_3'],
                b=self.get_bias(vgg_layers, 14))
        net['conv3_4'] = self.conv_layer('conv3_4', net['relu3_3'],
                W=self.get_weights(vgg_layers, 16))
        net['relu3_4'] = self.relu_layer('relu3_4', net['conv3_4'],
                b=self.get_bias(vgg_layers, 16))
        net['pool3'] = self.pool_layer('pool3', net['relu3_4'], self.parser)

          # layer4

        net['conv4_1'] = self.conv_layer('conv4_1', net['pool3'],
                W=self.get_weights(vgg_layers, 19))
        net['relu4_1'] = self.relu_layer('relu4_1', net['conv4_1'],
                b=self.get_bias(vgg_layers, 19))
        net['conv4_2'] = self.conv_layer('conv4_2', net['relu4_1'],
                W=self.get_weights(vgg_layers, 21))
        net['relu4_2'] = self.relu_layer('relu4_2', net['conv4_2'],
                b=self.get_bias(vgg_layers, 21))
        net['conv4_3'] = self.conv_layer('conv4_3', net['relu4_2'],
                W=self.get_weights(vgg_layers, 23))
        net['relu4_3'] = self.relu_layer('relu4_3', net['conv4_3'],
                b=self.get_bias(vgg_layers, 23))
        net['conv4_4'] = self.conv_layer('conv4_4', net['relu4_3'],
                W=self.get_weights(vgg_layers, 25))
        net['relu4_4'] = self.relu_layer('relu4_4', net['conv4_4'],
                b=self.get_bias(vgg_layers, 25))
        net['pool4'] = self.pool_layer('pool4', net['relu4_4'], self.parser)

          # layer5

        net['conv5_1'] = self.conv_layer('conv5_1', net['pool4'],
                W=self.get_weights(vgg_layers, 28))
        net['relu5_1'] = self.relu_layer('relu5_1', net['conv5_1'],
                b=self.get_bias(vgg_layers, 28))
        net['conv5_2'] = self.conv_layer('conv5_2', net['relu5_1'],
                W=self.get_weights(vgg_layers, 30))
        net['relu5_2'] = self.relu_layer('relu5_2', net['conv5_2'],
                b=self.get_bias(vgg_layers, 30))
        net['conv5_3'] = self.conv_layer('conv5_3', net['relu5_2'],
                W=self.get_weights(vgg_layers, 32))
        net['relu5_3'] = self.relu_layer('relu5_3', net['conv5_3'],
                b=self.get_bias(vgg_layers, 32))
        net['conv5_4'] = self.conv_layer('conv5_4', net['relu5_3'],
                W=self.get_weights(vgg_layers, 34))
        net['relu5_4'] = self.relu_layer('relu5_4', net['conv5_4'],
                b=self.get_bias(vgg_layers, 34))
        net['pool5'] = self.pool_layer('pool5', net['relu5_4'], self.parser)

        return net
