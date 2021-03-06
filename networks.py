import numpy as np
import tensorflow as tf
from glob import glob
from vgg16 import VGG16
from utils import gram_matrix
from skimage.transform import resize
from skimage.io import imread, imshow
from dl_layers import ActivationLayer, InstanceNormalizationLayer, ConvolutionalLayer, ConvolutionalTransposeLayer


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
        self.first_fit = True

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
        self.style_branch_train_op = None

        # other attributes
        self.session = None
        self.reconstruct_losses = None
        self.style_branch_losses = None
        self.style_losses = None
        self.content_losses = None

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
            InstanceNormalizationLayer(),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=64, stride_horizontal=2, stride_vertical=2),
            InstanceNormalizationLayer(),
            ActivationLayer(tf.nn.relu),
            ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=128, stride_horizontal=2, stride_vertical=2),
            InstanceNormalizationLayer(),
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
        for encoder_layer in [layer for layer in self.encoder_layers[::-1] if isinstance(layer, ConvolutionalLayer)]:
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

            self.decoder_layers.append(InstanceNormalizationLayer())
            self.decoder_layers.append(ActivationLayer(tf.nn.relu))

        # we don't want instance normalization or activation in the last layer
        self.decoder_layers = self.decoder_layers[:-2]

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
                InstanceNormalizationLayer(),
                ActivationLayer(tf.nn.relu),
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=256, padding="SAME"),
                InstanceNormalizationLayer(),
                ActivationLayer(tf.nn.relu),
                ConvolutionalLayer(filter_h=3, filter_w=3, maps_out=self.encoder_layers[-1].output_shape[-1], padding="SAME"),
                InstanceNormalizationLayer(),
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
            if isinstance(layer, InstanceNormalizationLayer):
                z = layer.forward(z, is_training=True)
            else:
                z = layer.forward(z)

        return z

    def decode(self, x):
        z = x

        for layer in self.decoder_layers:
            if isinstance(layer, InstanceNormalizationLayer):
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
            if isinstance(layer, InstanceNormalizationLayer):
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

        z = tf.gather_nd(z, style_indices)

        return z

    def initialize_style_branch(self, c, s, style_indices):
        # apply the appropriate styles to the content images
        self.style_content_imgs_op = self.apply_styles(c, style_indices)  # self.forward_style_n(c, 0)

        # initialize the style loss
        self.initialize_style_loss(self.style_content_imgs_op, s)

        # initialize the content loss
        self.initialize_content_loss(self.style_content_imgs_op, c)

        # define the style branch loss
        self.style_branch_loss = self.style_loss + self.content_loss

    def initialize_style_loss(self, styled_c, s):
        styled_content_outputs = [
                tf.map_fn(gram_matrix, self.vgg16.forward_conv_output(x=styled_c, layer_n=i)) for i in
                [2, 4, 6, 9]
            ]
        style_outputs = [
                tf.map_fn(gram_matrix, self.vgg16.forward_conv_output(x=s, layer_n=i)) for i in
                [2, 4, 6, 9]
            ]

        self.style_loss = 0
        for content_grams, style_grams in zip(styled_content_outputs, style_outputs):
            self.style_loss += tf.reduce_mean(
                tf.square(content_grams - style_grams)
            )

            self.style_loss = self.style_loss * self.style_loss_param / len(styled_content_outputs)

    def initialize_content_loss(self, styled_c, c):
        self.content_loss = tf.reduce_mean(
            tf.square(
                self.vgg16.forward_conv_output(styled_c, self.content_layer_n) -
                self.vgg16.forward_conv_output(c, self.content_layer_n)
            )
        )

    def initialize_vgg16(self):
        self.vgg16 = VGG16(input_shape=self.img_shape, trainable=False, session=self.session)

    def initialize_operations(self):
        # encoder
        self.reconstruct_op = self.reconstruct(self.tfX)
        self.encoder_loss = tf.reduce_mean(tf.square(self.tfX - self.reconstruct_op))
        self.encoder_train_op = self.optimizer.minimize(self.encoder_loss)

        # style bank
        self.initialize_style_branch(self.tfX, self.tfS, self.tfStyleIndices)
        style_vars = []
        for style_n in self.style_bank:
            style_vars += [
                layer.w for layer in self.style_bank[style_n] if isinstance(layer, ConvolutionalLayer)
            ]

            style_vars += [
                layer.b for layer in self.style_bank[style_n] if isinstance(layer, ConvolutionalLayer)
            ]

        # style cost / gradient clipping
        gvs = self.optimizer.compute_gradients(self.style_branch_loss, var_list=style_vars)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.style_branch_train_op = self.optimizer.apply_gradients(capped_gvs)
        #self.style_branch_train_op = self.optimizer.minimize(self.style_branch_loss)

    def set_session(self, session=None):
        if session is None:
            self.session = tf.Session(config=self.config)
        else:
            self.session = session

    def fit(self, n_epochs, n_steps, batch_size, optimizer=None, print_step=20, resume_training=True):
        # set session and optimizer
        if self.first_fit or not resume_training:
            self.set_session()
            self.optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer()

            # create lists to hold loss
            self.reconstruct_losses = []
            self.style_branch_losses = []
            self.content_losses = []
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
            self.vgg16.load_weights("./weights/vgg16_weights.npz", self.session)

            self.first_fit = False

        # load the data into memory
        c = np.array([
            resize(
                imread(img_path),
                self.img_shape[1:]
            ) * 255 for img_path in self.content_imgs
        ])

        s = [
            imread(img_path) for img_path in self.style_imgs
        ]

        s = np.array([
            resize(
                e if e.shape[0] < e.shape[1] else np.transpose(e, [1, 0, 2]),
                self.content_shape[1:]
            ) * 255 for e in s
        ])

        # train
        for i in range(n_epochs):
            for j in range(n_steps):
                c_i = self.sample_train_batch(batch_size, "content")
                s_i = self.sample_train_batch(batch_size, "style")

                c_batch = c[c_i]
                s_batch = s[s_i]

                style_indices = np.array([
                    [style_n, k] for k, style_n in enumerate(s_i)
                ])

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

                self.style_branch_losses.append(
                    self.session.run(
                        self.style_branch_loss,
                        feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: [[0, 0]]}
                    )
                )

                self.style_losses.append(
                    self.session.run(
                        self.style_loss,
                        feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: [[0, 0]]}
                    )
                )

                self.content_losses.append(
                    self.session.run(
                        self.content_loss,
                        feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: [[0, 0]]}
                    )
                )

                print("Epoch: %d. Reconstruct loss: %.2f. Style branch loss: %.2f. Style loss: %.2f. "
                      "Content loss: %.2f. Style indices: %s" % (
                        i,
                        self.reconstruct_losses[-1],
                        self.style_branch_losses[-1],
                        self.style_losses[-1],
                        self.content_losses[-1],
                        str(style_indices)
                    )
                )


model = StyleBank(
    img_shape=(None, 500, 800, 3),
    content_shape=(None, 600, 800, 3),
    style_imgs_path="./style/",
    content_imgs_path="./content/",
    style_loss_param=0.01,
    content_layer_n=9
)

model.fit(
    n_epochs=100000,
    n_steps=3,
    batch_size=3,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    print_step=10
)

img = resize(imread("./content/The-Grand-Belgrade-Fortress-and-Park-Kalemegdan.jpg"), (500, 800, 3)) * 255
img = np.expand_dims(img, axis=0)
style_img = resize(imread("./style/picasso.jpg"), (600, 800, 3)) * 255
style_img = np.expand_dims(style_img, axis=0)

# using gather
imgr = model.session.run(
    model.style_content_imgs_op,
    feed_dict={
        model.tfX: img,
        model.tfStyleIndices: [[0, 0]]
    }
)
imgr = imgr[0] - imgr.min()
imgr = imgr / imgr.max()
imshow(imgr)

xr = model.session.run(model.reconstruct_op, feed_dict={model.tfX: img})
xr = xr - xr.min()
xr = xr / xr.max()
imshow(xr[0, ...])
