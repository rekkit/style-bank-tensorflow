import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
from skimage.io import imread, imshow
from dl_layers import ActivationLayer, BatchNormalizationLayer, ConvolutionalLayer, ConvolutionalTransposeLayer


class StyleBank(object):
    def __init__(self, img_shape, n_styles, style_imgs_path, content_imgs_path):
        self.img_shape = img_shape
        self.n_styles = n_styles
        self.style_imgs_path = style_imgs_path
        self.content_imgs_path = content_imgs_path
        self.optimizer = None

        # auto-encoder
        self.encoder_layers = []
        self.decoder_layers = []
        self.initialize_encoder()

        # style bank
        self.style_bank = {k: None for k in range(self.n_styles)}
        self.initialize_style_bank()

        # operations
        self.reconstruct_op = None
        self.encoder_loss = None
        self.encoder_train_op = None

        # other attributes
        self.session = None
        self.losses = None

        # initialize the placeholder
        self.tfX = tf.placeholder(shape=(None, *img_shape[1:]), dtype=tf.float32, name="tfX")

    def initialize_encoder(self):
        # initialize the encoder layers
        self.encoder_layers = [
            ConvolutionalLayer(filter_h=9, filter_w=9, maps_out=32, layer_id=0, stride_horizontal=1, stride_vertical=1),
            BatchNormalizationLayer(axes=[1, 2], beta=1),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=64, layer_id=1, stride_horizontal=2, stride_vertical=2),
            BatchNormalizationLayer(axes=[1, 2], beta=1),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=128, layer_id=2, stride_horizontal=2, stride_vertical=2),
            BatchNormalizationLayer(axes=[1, 2], beta=1),
            ActivationLayer(tf.nn.relu)
        ]

        # set the input shape of each of the encoder layers and initialize weights
        self.encoder_layers[0].append_input_shape(input_shape=self.img_shape)
        self.encoder_layers[0].initialize_weights()

        for i in range(len(self.encoder_layers) - 1):
            self.encoder_layers[i+1].append_input_shape(
                self.encoder_layers[i].output_shape
            )

            self.encoder_layers[i+1].initialize_weights()

        # use the encoder layers to initialize the decoder layers
        self.decoder_layers = []
        for i, encoder_layer in enumerate(
                [layer for layer in self.encoder_layers[::-1] if isinstance(layer, ConvolutionalLayer)]
        ):
            self.decoder_layers.append(
                ConvolutionalTransposeLayer(
                    filter_h=encoder_layer.filter_h,
                    filter_w=encoder_layer.filter_w,
                    maps_out=encoder_layer.maps_in,
                    layer_id=i,
                    stride_horizontal=encoder_layer.stride_horizontal,
                    stride_vertical=encoder_layer.stride_vertical,
                    output_shape=encoder_layer.input_shape
                )
            )

            self.decoder_layers.append(BatchNormalizationLayer(axes=[1, 2], beta=1))
            self.decoder_layers.append(ActivationLayer(tf.nn.relu))

        # set the input shape of each of the decoder layers
        self.decoder_layers[0].append_input_shape(self.encoder_layers[-1].output_shape)
        self.decoder_layers[0].initialize_weights()

        for i in range(len(self.decoder_layers) - 1):
            self.decoder_layers[i+1].append_input_shape(
                self.decoder_layers[i].output_shape
            )

            self.decoder_layers[i+1].initialize_weights()

        # trace
        for layer in self.encoder_layers:
            print("Layer: ", type(layer), "Input shape: ", layer.input_shape, "Output shape: ", layer.output_shape)
        for layer in self.decoder_layers:
            print("Layer: ", type(layer), "Input shape: ", layer.input_shape, "Output shape: ", layer.output_shape)

    def initialize_style_bank(self):
        for i in self.style_bank:
            self.style_bank[i] = [
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, layer_id=0, padding="SAME"),
                BatchNormalizationLayer(axes=[1, 2], beta=1),
                ActivationLayer(tf.nn.relu),
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, layer_id=0, padding="SAME"),
                BatchNormalizationLayer(axes=[1, 2], beta=1),
                ActivationLayer(tf.nn.relu),
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, layer_id=0, padding="SAME"),
                BatchNormalizationLayer(axes=[1, 2], beta=1),
                ActivationLayer(tf.nn.relu),
            ]

            # append the input shapes of each of the convolutional layers in the style bank
            self.style_bank[i][0].append_input_shape(self.encoder_layers[-1].output_shape)
            self.style_bank[i][0].initialize_weights()

            for j in range(len(self.style_bank[i]) - 1):
                self.style_bank[i][j+1].append_input_shape(
                    self.style_bank[i][j].output_shape
                )

                self.style_bank[i][j+1].initialize_weights()

    def encode(self, x):
        z = x

        for layer in self.encoder_layers:
            if isinstance(layer, BatchNormalizationLayer):
                z = layer.forward(z, is_training=True)
            else:
                z = layer.forward(z)

        return z

    def decode(self, x):
        z = x

        for layer in self.decoder_layers:
            if isinstance(layer, BatchNormalizationLayer):
                z = layer.forward(z, is_training=True)
            else:
                z = layer.forward(z)

        return z

    def reconstruct(self, x):
        z = self.encode(x)

        return self.decode(z)

    def initialize_operations(self):
        self.reconstruct_op = self.reconstruct(self.tfX)
        self.encoder_loss = tf.reduce_mean(tf.square(self.tfX - self.reconstruct_op))
        self.encoder_train_op = self.optimizer.minimize(self.encoder_loss)

    def set_session(self):
        self.session = tf.Session()

    def fit(self, x, n_epochs, batch_size, optimizer=None, print_step=20):
        # set session and optimizer
        self.set_session()
        self.optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer()

        # define the number of steps we need
        n_steps = x.shape[0] // batch_size

        # create lists to hold loss
        self.losses = []

        # initialize operations
        self.initialize_operations()

        # initialize TF variables
        self.session.run(tf.global_variables_initializer())

        # train
        for i in range(n_epochs):
            x = shuffle(x)

            for j in range(n_steps):
                x_batch = x[j*batch_size: (j+1)*batch_size]

                self.session.run(
                    self.encoder_train_op,
                    feed_dict={self.tfX: x_batch}
                )

                if j % print_step == 0:
                    self.losses.append(
                        self.session.run(self.encoder_loss, feed_dict={self.tfX: x})
                    )

                    print("Epoch: %d. Step: %d. Loss: %.2f." % (i, j, self.losses[-1]))

            if x.shape[0] % batch_size > 0:
                self.session.run(
                    self.encoder_train_op,
                    feed_dict={self.tfX: x[(j+1)*batch_size:]}
                )


model = StyleBank(
    img_shape=(1, 459, 850, 3),
    n_styles=None,
    style_imgs_path=None,
    content_imgs_path=None
)

x = Image.open("./content/Van_Gogh_-_Starry_Night.jpg")
x = np.array(x)

# x = np.array(x.resize((850, 459)))
x = np.expand_dims(x, axis=0)

model.fit(
    x,
    n_epochs=2000,
    batch_size=1
)

xr = model.session.run(model.reconstruct_op, feed_dict={model.tfX: x})
xr = xr / xr.max()
imshow(xr[0, ...])
