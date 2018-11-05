import numpy as np
import tensorflow as tf
from dl_layers import ActivationLayer, ConvolutionalLayer, HiddenLayer, MaxPoolLayer


class VGG16:
    def __init__(self, input_shape, trainable=True, session=None):
        self.trainable = trainable
        self.session = session
        self.weights = None
        self.params = []
        self.BGR_MEAN_PIXELS = np.array([103.939, 116.779, 123.68]).reshape((1, 1, 1, 3)).astype(np.float32)

        # VGG16 architecture
        self.layers = [
            # block 1
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=64, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=64, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            MaxPoolLayer(window_h=2, window_w=2, stride_horizontal=2, stride_vertical=2),
            # block 2
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=128, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=128, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            MaxPoolLayer(window_h=2, window_w=2, stride_horizontal=2, stride_vertical=2),
            # block 3
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            MaxPoolLayer(window_h=2, window_w=2, stride_horizontal=2, stride_vertical=2),
            # block 4
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=512, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=512, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=512, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            MaxPoolLayer(window_h=2, window_w=2, stride_horizontal=2, stride_vertical=2),
            # block 5
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=512, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=512, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=512, trainable=self.trainable, padding="SAME"),
            ActivationLayer(tf.nn.relu),
            MaxPoolLayer(window_h=2, window_w=2, stride_horizontal=2, stride_vertical=2),
        ]

        self.layers[0].append_input_shape(input_shape=input_shape)
        self.layers[0].initialize_weights(0)
        self.params += [self.layers[0].w, self.layers[0].b]
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].append_input_shape(
                self.layers[i].output_shape
            )

            self.layers[i + 1].initialize_weights(i + 1)

            if isinstance(self.layers[i + 1], ConvolutionalLayer) or isinstance(self.layers[i + 1], HiddenLayer):
                self.params += [self.layers[i + 1].w, self.layers[i + 1].b]

    def load_weights(self, weights_path, session=None):
        if self.session is None:
            self.session = session

        if self.session is None:
            raise ValueError(
                "The session can not be None if you want to load weights."
            )

        self.weights = np.load(weights_path)
        keys = sorted(self.weights.keys())
        for i, k in enumerate(keys):
            if i == len(self.params):
                break
            print(i, k, np.shape(self.weights[k]))
            self.session.run(
                self.params[i].assign(
                    self.weights[k][:, :, ::-1, :] if (k.endswith("W") and i == 0) else self.weights[k]
                )
            )

    def forward_conv_output(self, x, layer_n, after_activation=True):
        if layer_n < 1 or layer_n > 13:
            raise ValueError(
                "The VGG16 neural network has 13 convolution layers. Therefore the parameter 'layer' must be between "
                "1 and 13."
            )

        i = 0
        z = x[:, :, :, ::-1] - self.BGR_MEAN_PIXELS
        for j in range(len(self.layers)):
            z = self.layers[j].forward(z)
            if isinstance(self.layers[j], ConvolutionalLayer):
                i += 1
                if i == layer_n:
                    if after_activation:
                        z = self.layers[j+1].forward(z)
                    return z
