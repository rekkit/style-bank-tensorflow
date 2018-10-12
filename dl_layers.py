import numpy as np
import tensorflow as tf


class ActivationLayer(object):
    with tf.variable_scope("activation_layer"):
        def __init__(self, activation_fn):
            self.activation_fn = activation_fn
            self.input_shape = None
            self.output_shape = None

        def append_input_shape(self, input_shape):
            self.input_shape = input_shape
            self.output_shape = input_shape

        def initialize_weights(self):
            pass

        def forward(self, x):
            return self.activation_fn(x)


class HiddenLayer(object):
    with tf.variable_scope("hidden_layer"):
        def __init__(self, n_in, n_out, layer_id, activation_fn):
            self.n_in = n_in
            self.n_out = n_out
            self.layer_id = layer_id
            self.activation_fn = activation_fn

            # create initializer and initialize weights and biases
            initializer = tf.contrib.layers.xavier_initializer(
                uniform=False,
                dtype=tf.float32
            )

            self.w = tf.Variable(
                initializer((n_in, n_out)),
                name="w_%d" % self.layer_id
            )

            self.b = tf.Variable(
                np.zeros(self.n_out, dtype=np.float32),
                name="b_%d" % self.layer_id
            )

        def forward_logits(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after multiplying the input by the weights and adding the biases. These are the values
                     that are fed into the activation function.
            """
            return tf.matmul(x, self.w) + self.b

        def forward(self, x):
            """
            :param x: The input to the hidden layer.
            :return: The values after performing forward propagation in this layer.
            """
            return self.activation_fn(
                self.forward_logits(x)
            )


class BatchNormalizationLayer(object):
    def __init__(self, axes=[0, 1, 2], beta=0.9):
        self.n_channels = None
        self.mean = None
        self.var = None
        self.offset = None
        self.scale = None
        self.axes = axes
        self.beta = beta
        self.input_shape = None
        self.output_shape = None

    def append_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

        # now that we have the input shape, we know how many channels the input has
        self.n_channels = self.input_shape[-1]
        self.mean = tf.Variable(np.zeros(self.n_channels), dtype=tf.float32, trainable=False)
        self.var = tf.Variable(np.ones(self.n_channels), dtype=tf.float32, trainable=False)
        self.offset = tf.Variable(np.zeros(self.n_channels), dtype=tf.float32)
        self.scale = tf.Variable(np.ones(self.n_channels), dtype=tf.float32)

    def initialize_weights(self):
        pass

    def forward(self, x, is_training):
        if is_training:
            # calculate the batch mean and variance
            batch_mean, batch_var = tf.nn.moments(
                x,
                axes=self.axes
            )

            # calculate the exponential weighted moving average of the mean and variance
            self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
            self.var = self.beta * self.var + (1 - self.beta) * batch_var

        # normalize the input
        return tf.nn.batch_normalization(
            x,
            self.mean,
            self.var,
            self.offset,
            self.scale,
            variance_epsilon=0.00001
        )


class ConvolutionalLayer(object):
    def __init__(
            self,
            filter_h,
            filter_w,
            maps_out,
            layer_id,
            with_bias=True,
            activation_fn=tf.nn.relu,
            stride_horizontal=1,
            stride_vertical=1,
            padding="VALID"
    ):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.maps_in = None
        self.maps_out = maps_out
        self.layer_id = layer_id
        self.with_bias = with_bias
        self.activation_fn = activation_fn
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.input_shape = None
        self.output_shape = None

        # weights
        self.w = None
        self.b = None

    def initialize_weights(self):
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
            name="conv_w_%d" % self.layer_id
        )

        if self.with_bias:
            self.b = tf.Variable(
                np.zeros(self.maps_out),
                dtype=tf.float32,
                name="conv_b_%d" % self.layer_id
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
            )

            output_w = int(
                (self.input_shape[2] - self.filter_w) / self.stride_horizontal + 1
            )

            self.output_shape = [self.input_shape[0], output_h, output_w, self.maps_out]

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

    # def forward(self, x):
    #     return self.activation_fn(
    #         self.forward_logits(x)
    #     )


class ConvolutionalTransposeLayer(object):
    def __init__(
            self,
            filter_h,
            filter_w,
            maps_out,
            layer_id,
            with_bias=True,
            activation_fn=tf.nn.relu,
            stride_horizontal=1,
            stride_vertical=1,
            padding="VALID",
            output_shape=None
    ):
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.maps_in = None
        self.maps_out = maps_out
        self.layer_id = layer_id
        self.with_bias = with_bias
        self.activation_fn = activation_fn
        self.stride_horizontal = stride_horizontal
        self.stride_vertical = stride_vertical
        self.padding = padding
        self.input_shape = None
        self.output_shape = output_shape

        # weights
        self.w = None
        self.b = None

    def initialize_weights(self):
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
            name="convt_w_%d" % self.layer_id
        )

        if self.with_bias:
            self.b = tf.Variable(
                np.zeros(self.maps_out),
                dtype=tf.float32,
                name="convt_b_%d" % self.layer_id
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
            output_shape=self.output_shape,
            strides=[1, self.stride_vertical, self.stride_horizontal, 1],
            padding=self.padding
        )

        if self.with_bias:
            return z + self.b

        return z

    # def forward(self, x):
    #     return self.activation_fn(
    #         self.forward_logits(x)
    #     )
