import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, UpSampling2D,
    concatenate, Reshape, Layer
)
from tensorflow.keras.models import Model

# Utility functions (reused from original TrackNetV4)
def rearrange_tensor(input_tensor, order):
    """
    Rearranges the dimensions of a tensor according to the specified order.
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    assert all([dim in order for dim in "BCHWT"]), "Order must contain all of BCHWT"
    return tf.transpose(input_tensor, [order.index(dim) for dim in "BTCHW"])

def reverse_rearrange_tensor(input_tensor, order):
    """
    Reverses the rearrangement of a tensor to its original order.
    """
    order = order.upper()
    assert len(set(order)) == 5, "Order must be a 5 unique character string"
    return tf.transpose(input_tensor, ["BTCHW".index(dim) for dim in order])

def power_normalization(input, a, b):
    """
    Power normalization function for attention map generation.
    """
    return 1 / (1 + tf.exp(-(5 / (0.45 * tf.abs(tf.tanh(a)) + 1e-1)) * (tf.abs(input) - 0.6 * tf.tanh(b))))

# MotionPromptLayer (reused with minor optimization)
class MotionPromptLayer(Layer):
    """
    A custom Keras layer for generating attention maps from video sequences.
    """
    def __init__(self, penalty_weight=0.0, **kwargs):
        super(MotionPromptLayer, self).__init__(**kwargs)
        self.input_permutation = "BTCHW"
        self.input_color_order = "RGB"
        self.color_map = {'R': 0, 'G': 1, 'B': 2}
        self.gray_scale = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)
        self.pn = power_normalization
        self.a = self.add_weight(shape=(), initializer=tf.constant_initializer(0.1), trainable=True, name='a')
        self.b = self.add_weight(shape=(), initializer=tf.constant_initializer(0.0), trainable=True, name='b')
        self.lambda1 = penalty_weight

    def call(self, video_seq):
        loss = tf.constant(0.0)
        video_seq = rearrange_tensor(video_seq, self.input_permutation)
        norm_seq = video_seq * 0.225 + 0.45
        idx_list = [self.color_map[idx] for idx in self.input_color_order]
        gray_scale_tensor = tf.gather(self.gray_scale, idx_list)
        weights = tf.cast(gray_scale_tensor, dtype=norm_seq.dtype)
        grayscale_video_seq = tf.einsum("btcwh,c->btwh", norm_seq, weights)
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]
        attention_map = self.pn(frame_diff, self.a, self.b)
        norm_attention = tf.expand_dims(attention_map, axis=2)

        if self.trainable:
            B, T, H, W = grayscale_video_seq.shape
            if B is None:
                B = 1
            temp_diff = norm_attention[:, 1:] - norm_attention[:, :-1]
            temporal_loss = tf.reduce_sum(tf.square(temp_diff)) / (H * W * (T - 2) * B)
            loss = self.lambda1 * temporal_loss
            self.add_loss(loss)

        return attention_map, loss

# FusionLayerTypeA (reused for simplicity)
class FusionLayerTypeA(Layer):
    """
    A Keras layer that incorporates motion using attention maps - version 1.
    """
    def call(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :]
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
        return tf.stack([output_1, output_2, output_3], axis=1)

def TrackNetV4Nano(input_height, input_width, fusion_layer_type="TypeA"):
    """
    Builds the TrackNetV4Nano model, a lightweight version of TrackNetV4 for improved performance.

    Args:
        input_height (int): The height of the input.
        input_width (int): The width of the input.

    Returns:
        Model: A Keras model instance.
    """
    fusion_layer = FusionLayerTypeA()

    # Input: 3 RGB frames (9 channels total)
    imgs_input = Input(shape=(9, input_height, input_width))
    motion_input = Reshape((3, 3, input_height, input_width))(imgs_input)

    # Motion prompt layer
    residual_maps, _ = MotionPromptLayer()(motion_input)

    # Encoder: Reduced layers and filters
    # Layer 1
    x = Conv2D(32, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(imgs_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 2
    x1 = Conv2D(32, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)

    # Layer 3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x1)

    # Layer 4
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x2 = BatchNormalization()(x)

    # Layer 5
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x2)

    # Layer 6
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Decoder: Reduced layers
    # Layer 7
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x2], axis=1)

    # Layer 8
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 9
    x = concatenate([UpSampling2D((2, 2), data_format='channels_first')(x), x1], axis=1)

    # Layer 10
    x = Conv2D(32, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Output layer
    x = Conv2D(3, (1, 1), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = fusion_layer([x, residual_maps])
    x = Activation('sigmoid')(x)

    # Model creation
    model = Model(inputs=imgs_input, outputs=x)
    return model