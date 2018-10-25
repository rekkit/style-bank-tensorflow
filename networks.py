import numpy as np
import tensorflow as tf
from glob import glob
from vgg16 import VGG16
from skimage.io import imread, imshow
from dl_layers import ActivationLayer, BatchNormalizationLayer, ConvolutionalLayer, ConvolutionalTransposeLayer


class StyleBank(object):
    def __init__(self, img_shape, content_shape, style_imgs_path, content_imgs_path, style_loss_param, content_layer_n):
        self.img_shape = img_shape
        self.content_shape = content_shape
        self.style_imgs = glob(style_imgs_path + '*')
        self.content_imgs = glob(content_imgs_path + '*')
        self.n_styles = len(self.style_imgs)
        self.n_content_imgs = len(self.content_imgs)
        self.style_loss_param = style_loss_param
        self.content_layer_n = content_layer_n
        self.optimizer = None
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # auto-encoder
        self.encoder_layers = []
        self.decoder_layers = []

        # style bank
        self.style_bank = {k: None for k in range(self.n_styles)}

        # operations
        self.reconstruct_op = None
        self.encoder_loss = None
        self.encoder_train_op = None
        self.style_loss = None
        self.content_loss = None
        self.style_banks = None
        self.style_branch_train_op = None

        # other attributes
        self.session = None
        self.reconstruct_losses = None
        self.style_losses = None

        # placeholders - we need one for content and one for style
        self.tfX = None
        self.tfS = None
        self.tfStyleIndices = None
        self.vgg16 = None

    def initialize_placeholders(self):
        # initialize the content and style image tensors
        self.tfX = tf.placeholder(shape=(None, *self.img_shape[1:]), dtype=tf.float32, name="tfX")
        self.tfS = tf.placeholder(shape=(None, *self.content_shape[1:]), dtype=tf.float32, name="tfS")
        self.tfStyleIndices = tf.placeholder(shape=(None, 2), dtype=tf.int32, name="tfStyleIndices")

    def initialize_encoder(self):
        # initialize the encoder layers
        self.encoder_layers = [
            ConvolutionalLayer(filter_h=9, filter_w=9, maps_out=32, stride_horizontal=1, stride_vertical=1),
            BatchNormalizationLayer(axes=[1, 2], beta=1),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=64, stride_horizontal=2, stride_vertical=2),
            BatchNormalizationLayer(axes=[1, 2], beta=1),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=128, stride_horizontal=2, stride_vertical=2),
            BatchNormalizationLayer(axes=[1, 2], beta=1),
            ActivationLayer(tf.nn.relu)
        ]

        # set the input shape of each of the encoder layers and initialize weights
        self.encoder_layers[0].append_input_shape(input_shape=self.img_shape)
        self.encoder_layers[0].initialize_weights(0)

        for i in range(len(self.encoder_layers) - 1):
            self.encoder_layers[i+1].append_input_shape(
                self.encoder_layers[i].output_shape
            )

            self.encoder_layers[i+1].initialize_weights(i+1)

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
                    stride_horizontal=encoder_layer.stride_horizontal,
                    stride_vertical=encoder_layer.stride_vertical,
                    output_shape=encoder_layer.input_shape
                )
            )

            self.decoder_layers.append(BatchNormalizationLayer(axes=[1, 2], beta=1))
            self.decoder_layers.append(ActivationLayer(tf.nn.relu))

        # set the input shape of each of the decoder layers
        self.decoder_layers[0].append_input_shape(self.encoder_layers[-1].output_shape)
        self.decoder_layers[0].initialize_weights(0)

        for i in range(len(self.decoder_layers) - 1):
            self.decoder_layers[i+1].append_input_shape(
                self.decoder_layers[i].output_shape
            )

            self.decoder_layers[i+1].initialize_weights(i+1)

        # trace
        for layer in self.encoder_layers:
            print("Layer: ", type(layer), "Input shape: ", layer.input_shape, "Output shape: ", layer.output_shape)
        for layer in self.decoder_layers:
            print("Layer: ", type(layer), "Input shape: ", layer.input_shape, "Output shape: ", layer.output_shape)

    def initialize_style_bank(self):
        for i in self.style_bank:
            self.style_bank[i] = [
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, padding="SAME"),
                BatchNormalizationLayer(axes=[1, 2], beta=1),
                ActivationLayer(tf.nn.relu),
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, padding="SAME"),
                BatchNormalizationLayer(axes=[1, 2], beta=1),
                ActivationLayer(tf.nn.relu),
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=self.encoder_layers[-1].output_shape[-1], padding="SAME"),
                BatchNormalizationLayer(axes=[1, 2], beta=1),
                ActivationLayer(tf.nn.relu)
            ]

            # append the input shapes of each of the convolutional layers in the style bank
            self.style_bank[i][0].append_input_shape(self.encoder_layers[-1].output_shape)
            self.style_bank[i][0].initialize_weights(0)

            for j in range(len(self.style_bank[i]) - 1):
                self.style_bank[i][j+1].append_input_shape(
                    self.style_bank[i][j].output_shape
                )

                self.style_bank[i][j+1].initialize_weights(j+1)

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

    def forward_style_n(self, x, style_n):
        z = self.encode(x)

        for layer in self.style_bank[style_n]:
            if isinstance(layer, BatchNormalizationLayer):
                z = layer.forward(z, is_training=True)
            else:
                z = layer.forward(z)

        z = self.decode(z)

        return z

    def sample_train_batch(self, batch_size, img_set):
        if img_set == "content":
            return np.random.choice(np.arange(0, self.n_content_imgs), size=batch_size, replace=False)

        elif img_set == "style":
            return np.random.choice(np.arange(0, self.n_styles), size=batch_size, replace=False)
        else:
            raise ValueError(
                "Expected the 'img_set' parameter to be either 'content' or 'style', received '%s' instead." % img_set
            )

    def apply_styles(self, c, style_indices):
        z = tf.stack(
            [self.forward_style_n(c, style_n) for style_n in range(self.n_styles)],
            axis=0
        )

        return tf.gather_nd(z, style_indices)

    def initialize_style_branch(self, c, s, style_indices):
        # apply the appropriate styles to the content images
        self.style_content_imgs_op = self.apply_styles(c, style_indices)

        # initialize the style loss
        self.initialize_style_loss(self.style_content_imgs_op, s)

        # initialize the content loss
        self.initialize_content_loss(self.style_content_imgs_op, c)

        # define the style branch loss
        self.style_branch_loss = self.content_loss + self.style_loss_param * self.style_loss

    def initialize_style_loss(self, styled_c, s):
        styled_content_outputs = [
                tf.map_fn(self.gram_matrix, self.vgg16.forward_conv_output(x=styled_c, layer_n=i)) for i in
                [2, 4, 7, 10, 13]
            ]
        style_outputs = [
                tf.map_fn(self.gram_matrix, self.vgg16.forward_conv_output(x=s, layer_n=i)) for i in
                [2, 4, 7, 10, 13]
            ]

        self.style_loss = 0
        for content_grams, style_grams in zip(styled_content_outputs, style_outputs):
            self.style_loss += tf.reduce_mean(
                tf.square(content_grams - style_grams)
            )

    def initialize_content_loss(self, styled_c, c):
        self.content_loss = tf.reduce_mean(
            tf.square(
                self.vgg16.forward_conv_output(styled_c, self.content_layer_n) -
                self.vgg16.forward_conv_output(c, self.content_layer_n)
            )
        )

    @staticmethod
    def gram_matrix(x):
        z = tf.reshape(x, [-1, tf.shape(x)[-1]])  # this makes z [H*W, C]
        z = tf.matmul(tf.transpose(z), z) / tf.cast(tf.reduce_prod(tf.shape(z)), dtype=tf.float32)

        return z

    def initialize_vgg16(self):
        self.vgg16 = VGG16(input_shape=self.img_shape, trainable=False, session=self.session)

    def initialize_operations(self):
        # encoder
        self.reconstruct_op = self.reconstruct(self.tfX)
        self.encoder_loss = tf.reduce_mean(tf.square(self.tfX - self.reconstruct_op))
        self.encoder_train_op = self.optimizer.minimize(self.encoder_loss)

        # style bank
        self.initialize_style_branch(self.tfX, self.tfS, self.tfStyleIndices)
        self.style_branch_train_op = self.optimizer.minimize(self.style_branch_loss)

    def set_session(self, session=None):
        if session is None:
            self.session = tf.Session(config=self.config)
        else:
            self.session = session

    def fit(self, n_epochs, n_steps, batch_size, optimizer=None, print_step=20):
        # set session and optimizer
        self.set_session()
        self.optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer()

        # create lists to hold loss
        self.reconstruct_losses = []
        self.style_losses = []

        # initialize the auto-encoder and the style bank
        self.initialize_encoder()
        self.initialize_style_bank()

        # initialize operations
        self.initialize_placeholders()
        self.initialize_vgg16()
        self.initialize_operations()

        # initialize TF variables
        self.session.run(tf.global_variables_initializer())

        # load the data into memory
        c = np.array([
            imread(img_path) for img_path in self.content_imgs
        ])

        s = np.array([
            imread(img_path) for img_path in self.style_imgs
        ])

        # train
        for i in range(n_epochs):
            for j in range(n_steps):
                c_i = self.sample_train_batch(batch_size, "content")
                s_i = self.sample_train_batch(batch_size, "style")

                c_batch = c[c_i]
                s_batch = s[s_i]
                style_indices = [
                    [style_n, k] for k, style_n in enumerate(s_i)
                ]

                # style branch training
                self.session.run(
                    self.style_branch_train_op,
                    feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: style_indices}
                )

            c_i = self.sample_train_batch(batch_size, "content")
            c_batch = c[c_i]
            self.session.run(
                self.encoder_train_op,
                feed_dict={self.tfX: c_batch}
            )

            if i > 0 and i % print_step == 0:
                self.reconstruct_losses.append(
                    self.session.run(
                        self.encoder_loss,
                        feed_dict={self.tfX: c_batch}
                    )
                )

                self.style_losses.append(
                    self.session.run(
                        self.style_loss,
                        feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: style_indices}
                    )
                )

                print("Epoch: %d. Reconstruct loss: %.2f. Style loss: %.2f." % (
                        i, self.reconstruct_losses[-1], self.style_losses[-1]
                    )
                )


model = StyleBank(
    img_shape=(1, 459, 850, 3),
    content_shape=(1, 1014, 1280, 3),
    style_imgs_path="./style/",
    content_imgs_path="./content/",
    style_loss_param=1000,
    content_layer_n=5
)

model.fit(
    n_epochs=400,
    n_steps=3,
    batch_size=1
)

img = imread("./content/The-Grand-Belgrade-Fortress-and-Park-Kalemegdan.jpg")
imgr = model.session.run(
    model.style_content_imgs_op,
    feed_dict={model.tfX: np.expand_dims(img, axis=0), model.tfStyleIndices: [[0, 0]]}
)
imgr = imgr[0] / imgr.max()
imshow(imgr)

# xr = model.session.run(model.reconstruct_op, feed_dict={model.tfX: x})
# xr = xr / xr.max()
# imshow(xr[0, ...])
