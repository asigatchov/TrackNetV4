# import tensorflow as tf
from models.TrackNetV4Small import MotionPromptLayer, FusionLayerTypeA
from util import custom_loss
# import tf2onnx
# # Define custom objects
# custom_objects = {
#     "MotionPromptLayer": MotionPromptLayer,
#     "FusionLayerTypeA": FusionLayerTypeA,
#     "custom_loss": custom_loss
# }

# # Load the Keras model
# model = tf.keras.models.load_model(
#     "./models/track_netv4_small_30e_badminton.keras",
#  #   "track_netv4_saved_model.h5",
#     custom_objects=custom_objects
# )

# # Define the serving signature with the correct input shape
# @tf.function(input_signature=[tf.TensorSpec(shape=[None, 9, 288, 512], dtype=tf.float32, name="input")])
# def serving_fn(input_tensor):
#     return {"output": model(input_tensor, training=False)}

# # Save the model as SavedModel with the signature


# # Convert to ONNX
# onnx_model, _ = tf2onnx.convert.from_keras(
#     model,
#     input_signature=[tf.TensorSpec([None, 9, 288, 512], tf.float32, name="input")],
#     output_path="track_netv4.onnx"
# )


import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import Layer

model_name = "./models/track_netv4_middle_13e_badminton.keras"
# Define custom objects
custom_objects = {
    "MotionPromptLayer": MotionPromptLayer,
    "FusionLayerTypeA": FusionLayerTypeA,
    "custom_loss": custom_loss
}

# Load the trained model
model = tf.keras.models.load_model(
    #"./models/track_netv4_small_30e_badminton.keras",
    model_name,
    custom_objects=custom_objects
)


model_onnx = model_name.replace('keras', 'onnx')

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec([None, 9, 288, 512], tf.float32, name="input")],
    opset=13,  # Use a high opset for better compatibility
    output_path=model_onnx,
)


# Define serving signature
# @tf.function(input_signature=[tf.TensorSpec([None, 9, 288, 512], tf.float32, name="input")])
# def serving_fn(input_tensor):
#     return {"output": model(input_tensor, training=False)}

# # Save as SavedModel
# tf.saved_model.save(
#     model,
#     "track_netv4_saved_model",
#     signatures={"serving_default": serving_fn}
# )

# # Convert to OpenVINO IR
# import openvino as ov
# ov_model = ov.convert_model("track_netv4_saved_model", input=[[1,9,288,512]])
# ov.save_model(ov_model, "track_netv4_ir/model.xml")
