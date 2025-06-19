import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model("./models/track_netv4_small_30e_badminton.keras")

# Set learning phase to 0 (inference mode)
tf.keras.backend.set_learning_phase(0)

# Save as TensorFlow SavedModel
model.save("track_netv4_saved_model")
