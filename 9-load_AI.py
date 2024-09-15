import tensorflow as tf
import hickle as hkl

# Load the model
model = tf.keras.models.load_model('fencing_AI_checkpoint.h5')

# Test the model
test_set = hkl.load('test_data.hkl')
test_labels = hkl.load('test_lbls.hkl')

# Evaluate the mode
model.evaluate(test_set, test_labels)