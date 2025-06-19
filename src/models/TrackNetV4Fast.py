import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Reshape,
    Layer,
)
from tensorflow.keras.models import Model


# Simplified utility functions
def power_normalization(input, a, b):
    """
    Simplified power normalization for attention map generation.
    """
    return tf.sigmoid(
        (tf.abs(input) - 0.6 * tf.tanh(b)) * (5 / (0.45 * tf.abs(tf.tanh(a)) + 1e-5))
    )


# Optimized MotionPromptLayer with fixed grayscale computation
class MotionPromptLayer(Layer):
    """
    Simplified MotionPromptLayer for generating attention maps with reduced complexity.
    """

    def __init__(self, **kwargs):
        super(MotionPromptLayer, self).__init__(**kwargs)
        self.gray_scale = tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)
        self.a = self.add_weight(
            shape=(), initializer=tf.constant_initializer(0.1), trainable=True, name="a"
        )
        self.b = self.add_weight(
            shape=(), initializer=tf.constant_initializer(0.0), trainable=True, name="b"
        )

    def call(self, video_seq):
        # Reshape to (B, T, C, H, W) and normalize
        norm_seq = video_seq * 0.225 + 0.45
        # Reshape gray_scale to [1, 1, 3, 1, 1] for broadcasting across channels
        gray_scale_reshaped = tf.reshape(self.gray_scale, [1, 1, 3, 1, 1])
        # Compute grayscale: multiply norm_seq with gray_scale and sum along channel axis (axis 2)
        grayscale_video_seq = tf.reduce_sum(
            norm_seq * gray_scale_reshaped, axis=2, keepdims=False
        )
        # Compute frame differences
        frame_diff = grayscale_video_seq[:, 1:] - grayscale_video_seq[:, :-1]
        # Generate attention map
        attention_map = power_normalization(frame_diff, self.a, self.b)
        return attention_map

    def compute_output_shape(self, input_shape):
        # Input shape: [batch, time, channels, height, width]   
        # Output shape: [batch, time-1, height, width]
        return (input_shape[0], input_shape[1] - 1, input_shape[3], input_shape[4])


# Simplified FusionLayer
class FusionLayer(Layer):
    """
    Simplified fusion layer to combine feature maps with attention maps.
    """

    def call(self, inputs):
        feature_map, attention_map = inputs
        output_1 = feature_map[:, 0, :, :]
        output_2 = feature_map[:, 1, :, :] * attention_map[:, 0, :, :]
        output_3 = feature_map[:, 2, :, :] * attention_map[:, 1, :, :]
        return tf.stack([output_1, output_2, output_3], axis=1)

    def compute_output_shape(self, input_shape):
        # input_shape[0]: feature_map [batch, 3, height, width]
        # input_shape[1]: attention_map [batch, 2, height, width]
        return (input_shape[0][0], 3, input_shape[0][2], input_shape[0][3])


def TrackNetV4Fast(input_height, input_width):
    """
    Optimized TrackNetV4 model for OpenVINO INT8 inference on Intel GPU targeting 30+ FPS.

    Args:
        input_height (int): The height of the input.
        input_width (int): The width of the input.

    Returns:
        Model: A Keras model instance.
    """
    # Input: 3 RGB frames (9 channels total)
    imgs_input = Input(shape=(9, input_height, input_width))
    motion_input = Reshape((3, 3, input_height, input_width))(imgs_input)

    # Motion prompt layer
    residual_maps = MotionPromptLayer()(motion_input)

    # Encoder: Reduced layers and filters
    # Layer 1: Reduced filters to 16
    x = Conv2D(
        16,
        (3, 3),
        padding="same",
        data_format="channels_first",
        kernel_initializer="glorot_uniform",
    )(imgs_input)
    x = Activation("relu")(x)

    # Layer 2: Reduced filters to 16
    x1 = Conv2D(
        16,
        (3, 3),
        padding="same",
        data_format="channels_first",
        kernel_initializer="glorot_uniform",
    )(x)
    x1 = Activation("relu")(x1)

    # Layer 3: MaxPooling
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x1)

    # Layer 4: Reduced filters to 32
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        data_format="channels_first",
        kernel_initializer="glorot_uniform",
    )(x)
    x = Activation("relu")(x)
    x2 = x  # Skip connection

    # Layer 5: MaxPooling
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format="channels_first")(x2)

    # Layer 6: Reduced filters to 64
    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        data_format="channels_first",
        kernel_initializer="glorot_uniform",
    )(x)
    x = Activation("relu")(x)

    # Decoder
    # Layer 7: Upsampling + skip connection
    x = concatenate([UpSampling2D((2, 2), data_format="channels_first")(x), x2], axis=1)

    # Layer 8: Reduced filters to 32
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        data_format="channels_first",
        kernel_initializer="glorot_uniform",
    )(x)
    x = Activation("relu")(x)

    # Layer 9: Upsampling + skip connection
    x = concatenate([UpSampling2D((2, 2), data_format="channels_first")(x), x1], axis=1)

    # Layer 10: Reduced filters to 16
    x = Conv2D(
        16,
        (3, 3),
        padding="same",
        data_format="channels_first",
        kernel_initializer="glorot_uniform",
    )(x)
    x = Activation("relu")(x)

    # Output layer
    x = Conv2D(
        3,
        (1, 1),
        padding="same",
        data_format="channels_first",
        kernel_initializer="glorot_uniform",
    )(x)
    x = FusionLayer()([x, residual_maps])
    x = Activation("sigmoid")(x)

    # Model creation
    model = Model(inputs=imgs_input, outputs=x)
    return model
