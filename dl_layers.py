import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import instance_norm


class ActivationLayer:
    with tf.variable_scope("activation_layer"):
        def __init__(self, activation_fn):
            self.activation_fn = activation_fn
            self.input_shape = None
            self.output_shape = None

        def append_input_shape(self, input_shape):
            self.input_shape = input_shape
            self.output_shape = input_shape

        def initialize_weights(self, layer_id):
            pass

        def forward(self, x):
            return self.activation_fn(x)


class FlattenLayer:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def append_input_shape(self, input_shape):
        self.input_shape = input_shape
        if None in self.input_shape[1:]:
            self.output_shape = (self.input_shape[0], None)
        else:
            self.output_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

    def initialize_weights(self, layer_id):
        pass

    @staticmethod
    def forward(x):
        return tf.reshape(
            x,
            shape=(tf.shape(x)[0], -1)
        )


class MaxPoolLayer:
    def __init__(self, window_h, window_w, stride_horizontal, stride_vertical, padding="VALID"):
        self.window_h = window_h
        self.window_w = window_w
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.input_shape = None
        self.output_shape = None

    def append_input_shape(self, input_shape):
        self.input_shape = input_shape

        if self.padding == "SAME":
            self.output_shape = self.input_shape
        else:
            output_h = int(
                (self.input_shape[1] - self.window_h) / self.stride_vertical + 1
            ) if self.input_shape[1] is not None else None

            output_w = int(
                (self.input_shape[2] - self.window_w) / self.stride_horizontal + 1
            ) if self.input_shape[2] is not None else None

            self.output_shape = (self.input_shape[0], output_h, output_w, self.input_shape[-1])

    def initialize_weights(self, layer_id):
        pass

    def forward(self, x):
        return tf.nn.max_pool(
            x,
            ksize=[1, self.window_h, self.window_w, 1],
            strides=[1, self.stride_vertical, self.stride_horizontal, 1],
            padding=self.padding,
            name="max_pool"
        )


class HiddenLayer:
    with tf.variable_scope("hidden_layer"):
        def __init__(self, n_out, trainable=True):
            self.n_in = None
            self.n_out = n_out
            self.input_shape = None
            self.output_shape = None
            self.trainable = trainable
            self.w = None
            self.b = None

        def initialize_weights(self, layer_id):
            # create initializer and initialize weights and biases
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.w = tf.Variable(
                initializer((self.n_in, self.n_out)),
                name="w_%d" % layer_id,
                trainable=self.trainable
            )

            self.b = tf.Variable(
                np.zeros(self.n_out, dtype=np.float32),
                name="b_%d" % layer_id,
                trainable=self.trainable
            )

        def append_input_shape(self, input_shape):
            if len(input_shape) > 2:
                raise ValueError(
                    "Expected the input to the hidden layer to be two-dimensional but received %d dimensions." %
                    len(input_shape)
                )

            self.input_shape = input_shape
            self.n_in = self.input_shape[1]
            self.output_shape = (input_shape[0], self.n_out)

        def forward(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after multiplying the input by the weights and adding the biases. These are the values
                     that are fed into the activation function.
            """
            return tf.matmul(x, self.w) + self.b


class InstanceNormalizationLayer:
    def __init__(self, trainable=True):
        self.input_shape = None
        self.output_shape = None
        self.trainable = trainable

    def append_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def initialize_weights(self, layer_id):
        pass

    @staticmethod
    def forward(x, is_training):
        return instance_norm(
            x,
            center=True,
            scale=True,
            epsilon=1e-06,
            trainable=is_training
        )


class BatchNormalizationLayer:
    def __init__(self, axes=[0, 1, 2], beta=0.1, trainable=True):
        self.n_channels = None
        self.mean = None
        self.var = None
        self.offset = None
        self.scale = None
        self.axes = axes
        self.beta = beta
        self.input_shape = None
        self.output_shape = None
        self.trainable = trainable

    def append_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

        # now that we have the input shape, we know how many channels the input has
        self.n_channels = self.input_shape[-1]
        self.mean = tf.Variable(np.zeros(self.n_channels), dtype=tf.float32, trainable=False)
        self.var = tf.Variable(np.ones(self.n_channels), dtype=tf.float32, trainable=False)
        self.offset = tf.Variable(np.zeros(self.n_channels), dtype=tf.float32, trainable=self.trainable)
        self.scale = tf.Variable(np.ones(self.n_channels), dtype=tf.float32, trainable=self.trainable)

    def initialize_weights(self, layer_id):
        pass

    def forward(self, x, is_training):
        if is_training:
            # calculate the batch mean and variance
            batch_mean, batch_var = tf.nn.moments(
                x,
                axes=self.axes
            )

            # calculate the exponential weighted moving average of the mean and variance
            self.mean = (1 - self.beta) * self.mean + self.beta * batch_mean
            self.var = (1 - self.beta) * self.var + self.beta * batch_var

        # normalize the input
        return tf.nn.batch_normalization(
            x,
            self.mean,
            self.var,
            self.offset,
            self.scale,
            variance_epsilon=0.00001
        )


class ConvolutionalLayer:
    def __init__(
            self,
            filter_h,
            filter_w,
            maps_out,
            with_bias=True,
            stride_horizontal=1,
            stride_vertical=1,
            padding="VALID",
            trainable=True
    ):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.maps_in = None
        self.maps_out = maps_out
        self.with_bias = with_bias
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.input_shape = None
        self.output_shape = None
        self.trainable = trainable

        # weights
        self.w = None
        self.b = None

    def initialize_weights(self, layer_id):
        # first we have to check whether the number of input maps has been provided
        if self.maps_in is None:
            raise ValueError(
                "The number of input maps (maps_in) needs to be provided before initializing weights."
            )

        # create the weight initializer
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False,
            dtype=tf.float32
        )

        # print("filter_h: ", filter_h, "filter_w: ", filter_w, "maps_in: ", maps_in, "maps_out: ", maps_out)
        self.w = tf.Variable(
            initializer((self.filter_h, self.filter_w, self.maps_in, self.maps_out)),
            name="conv_w_%d" % layer_id,
            trainable=self.trainable
        )

        if self.with_bias:
            self.b = tf.Variable(
                np.zeros(self.maps_out),
                dtype=tf.float32,
                name="conv_b_%d" % layer_id,
                trainable=self.trainable
            )

    def append_input_shape(self, input_shape):
        # set the input shape and the number of input feature maps
        self.input_shape = input_shape
        self.maps_in = input_shape[-1]

        # since we now have the input shape, we can also calculate the shape of the output
        # this assumes NHWC format
        if self.padding == "SAME":
            self.output_shape = [*self.input_shape[:-1], self.maps_out]
        else:
            output_h = int(
                (self.input_shape[1] - self.filter_h) / self.stride_vertical + 1
            ) if self.input_shape[1] is not None else None

            output_w = int(
                (self.input_shape[2] - self.filter_w) / self.stride_horizontal + 1
            ) if self.input_shape[2] is not None else None

            self.output_shape = (self.input_shape[0], output_h, output_w, self.maps_out)

    def forward(self, x):
        z = tf.nn.conv2d(
            x,
            filter=self.w,
            strides=[1, self.stride_horizontal, self.stride_vertical, 1],
            padding=self.padding
        )

        if self.with_bias:
            return z + self.b

        return z


class ConvolutionalTransposeLayer:
    def __init__(
            self,
            filter_h,
            filter_w,
            maps_out,
            with_bias=True,
            stride_horizontal=1,
            stride_vertical=1,
            padding="VALID",
            output_shape=None,
            trainable=True
    ):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.maps_in = None
        self.maps_out = maps_out
        self.with_bias = with_bias
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.input_shape = None
        self.output_shape = output_shape
        self.trainable = trainable

        # weights
        self.w = None
        self.b = None

    def initialize_weights(self, layer_id):
        # first we have to check whether the number of input maps has been provided
        if self.maps_in is None:
            raise ValueError(
                "The number of input maps (maps_in) needs to be provided before initializing weights."
            )

        # create the weight initializer
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=False,
            dtype=tf.float32
        )

        # print("filter_h: ", filter_h, "filter_w: ", filter_w, "maps_in: ", maps_in, "maps_out: ", maps_out)
        self.w = tf.Variable(
            initializer((self.filter_h, self.filter_w, self.maps_out, self.maps_in)),
            name="convt_w_%d" % layer_id,
            trainable=self.trainable
        )

        if self.with_bias:
            self.b = tf.Variable(
                np.zeros(self.maps_out),
                dtype=tf.float32,
                name="convt_b_%d" % layer_id,
                trainable=self.trainable
            )

    def append_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.maps_in = input_shape[-1]

    def append_output_shape(self, output_shape):
        self.output_shape = output_shape

    def forward(self, x):
        if self.output_shape is None:
            raise ValueError(
                "You need to specify the output shape (output_shape) of the transpose convolution. You can do this "
                "either when you are initializing the class or by using the 'append_output_shape' method."
            )

        z = tf.nn.conv2d_transpose(
            x,
            filter=self.w,
            output_shape=(tf.shape(x)[0], *self.output_shape[1:]),
            strides=[1, self.stride_vertical, self.stride_horizontal, 1],
            padding=self.padding
        )

        if self.with_bias:
            return z + self.b

        return z
