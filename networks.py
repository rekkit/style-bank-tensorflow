import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from vgg16 import VGG16
from sklearn.utils import shuffle
from skimage.io import imread, imshow
from dl_layers import ActivationLayer, BatchNormalizationLayer, ConvolutionalLayer, ConvolutionalTransposeLayer


class StyleBank(object):
    def __init__(self, img_shape, style_imgs_path, content_imgs_path, style_loss_param):
        self.img_shape = img_shape
        self.style_imgs = glob(style_imgs_path + '*')
        self.content_imgs = glob(content_imgs_path + '*')
        self.n_styles = len(self.style_imgs)
        self.n_content_imgs = len(self.content_imgs)
        self.style_loss_param = style_loss_param
        self.optimizer = None

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

        # other attributes
        self.session = None
        self.losses = None

        # placeholders - we need one for content and one for style
        self.tfX = None
        self.tfS = None
        self.content_vgg = None
        self.style_vgg = None
        self.applied_styles = None

    def initialize_placeholders(self):
        # initialize the content and style image tensors
        self.tfX = tf.placeholder(shape=(None, *self.img_shape[1:]), dtype=tf.float32, name="tfX")
        self.tfS = tf.placeholder(shape=(None, *self.img_shape[1:]), dtype=tf.float32, name="tfS")

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
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=self.encoder_layers[-1].output_shape[-1], layer_id=0, padding="SAME"),
                BatchNormalizationLayer(axes=[1, 2], beta=1),
                ActivationLayer(tf.nn.relu)
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

    def apply_style(self, c, indices):
        return tf.concat(
            [self.forward_style_n(tf.expand_dims(c[i], axis=0), style_n) for i, style_n in enumerate(indices)],
            axis=0
        )

    def initialize_style_branch(self, c, s, style_indices, content_layer_n):
        # apply the appropriate styles to the content images
        z = self.apply_style(c, style_indices)
        self.styled_content_vgg = self.initialize_vgg16(tf.shape(z), z)

        # pass the content images through the VGG16 without applying any styles
        self.content_vgg = self.initialize_vgg16(tf.shape(z), c)

        # pass the style images through the VGG16
        self.style_vgg = self.initialize_vgg16(tf.shape(s), s)

        # initialize the style loss
        self.initialize_style_loss()

        # initialize the content loss
        self.initialize_content_loss(content_layer_n)

        # define the style branch loss
        self.style_branch_loss = self.content_loss + self.style_loss_param * self.style_loss

    def initialize_style_loss(self):
        styled_content_outputs = [
            tf.map_fn(self.gram_matrix, v) for k, v in self.styled_content_vgg.outputs.items() if k.endswith("conv2")
        ]
        style_outputs = [
            tf.map_fn(self.gram_matrix, v) for k, v in self.style_vgg.outputs.items() if k.endswith("conv2")
        ]

        self.style_loss = 0
        for content_grams, style_grams in zip(styled_content_outputs, style_outputs):
            self.style_loss += tf.square(content_grams - style_grams)

    def initialize_content_loss(self, content_layer_n):
        conv_output_names = sorted([k for k, _ in self.styled_content_vgg.outputs.items() if "conv" in k])
        layer_name = conv_output_names[content_layer_n]

        self.content_loss = tf.square(
            self.styled_content_vgg.outputs[layer_name] - self.content_vgg.outputs[layer_name]
        )

    @staticmethod
    def gram_matrix(x):
        z = tf.reshape(x, [-1, tf.shape(x)[-1]])  # this makes z [H*W, C]
        z = tf.matmul(tf.transpose(z), z) / tf.constant(tf.reduce_prod(tf.shape(z)))

        return z

    @staticmethod
    def initialize_vgg16(img_shape, input_tensor):
        return VGG16(img_shape, input_tensor)

    def initialize_operations(self):
        # encoder
        self.reconstruct_op = self.reconstruct(self.tfX)
        self.encoder_loss = tf.reduce_mean(tf.square(self.tfX - self.reconstruct_op))
        self.encoder_train_op = self.optimizer.minimize(self.encoder_loss)

        # style bank
        self.applied_styles = {i: self.apply_style(self.tfX, i) for i in range(self.n_styles)}

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

        # initialize the auto-encoder and the style bank
        self.initialize_encoder()
        self.initialize_style_bank()

        # initialize operations
        self.initialize_placeholders()
        self.initialize_operations()
        self.initialize_style_branch()

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
    style_imgs_path="./content/",
    content_imgs_path="./style/"
)

# x = Image.open("./style/Van_Gogh_-_Starry_Night.jpg")
# x = np.array(x)
#
# # x = np.array(x.resize((850, 459)))
# x = np.expand_dims(x, axis=0)
#
# model.fit(
#     x,
#     n_epochs=2000,
#     batch_size=1
# )

# xr = model.session.run(model.reconstruct_op, feed_dict={model.tfX: x})
# xr = xr / xr.max()
# imshow(xr[0, ...])
