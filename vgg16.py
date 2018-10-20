import numpy as np
import tensorflow as tf
import tempfile

RGB_MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3)).astype(np.float32)
DEFAULT_IMAGE_SHAPE = (None, 224, 224, 3)


class VGG16:
    """
    A class that builds a TF graph with a pre-trained VGG16 model (on imagenet)
    Also takes care of preprocessing. Input should be a regular RGB image (0-255)
    """
    def __init__(self, image_shape=DEFAULT_IMAGE_SHAPE, input_tensor=None):
        self.image_shape = image_shape
        self._build_graph(input_tensor)

    def _build_graph(self, input_tensor):
        with tf.Session() as sess:
            with tf.variable_scope('VGG16'):
                with tf.name_scope('inputs'):
                    if input_tensor is None:
                        input_tensor = tf.placeholder(tf.float32, shape=self.image_shape, name='input_img')
                    else:
                        pass  # assert self.image_shape == input_tensor.shape
                    self.input_tensor = input_tensor

                with tf.name_scope('preprocessing'):
                    img = self.input_tensor - RGB_MEAN_PIXELS
                    img = tf.reverse(img, axis=[-1])

                with tf.variable_scope('model'):
                    self.vgg16 = tf.contrib.keras.applications.VGG16(
                        weights='imagenet',
                        include_top=False,
                        input_tensor=img
                    )

                    for layer in self.vgg16:
                        layer.trainable = False

                self.outputs = {l.name: l.output for l in self.vgg16.layers}

            self.vgg_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VGG16/model')

            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(self.vgg_weights).save(sess, f.name)

        self.model_weights_tensors = set(self.vgg_weights)

    def load_weights(self):
        sess = tf.get_default_session()
        tf.train.Saver(self.vgg_weights).restore(sess, self.tf_checkpoint_path)

    def __getitem__(self, key):
        return self.outputs[key]
