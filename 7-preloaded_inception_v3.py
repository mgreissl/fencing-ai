
# Run a pretrained_inception net over the clips to get a 1*2048 conv feature vector for each frame, save these.
# thanks to https://github.com/oduerr/dl_tutorial/blob/master/tensorflow/inception_cifar10/cifar-10_experiment.py
# for the example code which this is inspired by.
import tensorflow as tf
import numpy as np
import os
import hickle as hkl
import time
import zipfile
import urllib.request
from PIL import Image
import io

model_dir = 'model/'  # Path to your model directory


# Function to download the pre-trained model
def maybe_download_and_extract(model_dir='model/'):
    """Download and extract pre-trained Inception model (ZIP file)."""
    url = 'http://download.tensorflow.org/models/inception_dec_2015.zip'
    model_file = os.path.join(model_dir, 'tensorflow_inception_graph.pb')
    zip_file = os.path.join(model_dir, 'inception_dec_2015.zip')

    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if model already exists
    if not os.path.exists(model_file):
        print("Downloading model...")
        urllib.request.urlretrieve(url, zip_file)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
            print(f"Model extracted to {model_dir}")

        # Clean up the ZIP file after extraction
        os.remove(zip_file)
    else:
        print(f"Model already exists at {model_file}")


# Function to load the model into TensorFlow's graph
def create_graph(model_dir='model/'):
    """Loads the Inception model (classify_image_graph_def.pb) into the TensorFlow graph."""
    with tf.io.gfile.GFile(os.path.join(model_dir, 'tensorflow_inception_graph.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


# Download and create the graph
maybe_download_and_extract(model_dir)
create_graph(model_dir)

# Disable eager execution for TensorFlow 1.x-style session
tf.compat.v1.disable_eager_execution()

# Start session and load the model into it
with tf.compat.v1.Session() as sess:
    # Debug: Print all operation names to find the correct tensor
    for op in sess.graph.get_operations():
        print(op.name)

    # Try to retrieve the correct tensor by name
    try:
        representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    except KeyError:
        print("Tensor 'pool_3:0' not found. Please check the printed operations to find the correct tensor name.")
        exit(1)

    # Iterate over files and process the data
    for i in os.listdir(os.getcwd() + "/preinception_data/"):
        if i.endswith(".hkl"):
            if "set" in i:
                number = i.split("-")[-1].replace(".hkl", "")
                print(number)

                train_set = hkl.load(os.getcwd() + '/preinception_data/' + i)
                print("Training Data:", train_set.shape)

                train_labels = hkl.load(os.getcwd() + '/final_training_data/' + "train_labels-" + str(number) + ".hkl")
                print("Train Labels:", train_labels.shape)

                for example in range(len(train_set)):
                    frame_representation = np.zeros((len(train_set[example]), 2048), dtype='float32')
                    start = time.time()

                    for frame in range(len(train_set[example])):
                        # Convert the NumPy array to a JPEG-encoded byte string
                        image_array = train_set[example][frame]  # This is the frame in shape (345, 640, 3)

                        # Convert NumPy array to PIL Image
                        image = Image.fromarray(image_array)

                        # Save the image to a bytes buffer in JPEG format
                        with io.BytesIO() as img_buffer:
                            image.save(img_buffer, format='JPEG')
                            jpeg_data = img_buffer.getvalue()  # This is the JPEG-encoded byte string

                        # Now run the session using the JPEG-encoded byte string
                        rep = sess.run(representation_tensor, {'DecodeJpeg/contents:0': jpeg_data})
                        frame_representation[frame] = np.squeeze(rep)

                    frame_representation = np.expand_dims(frame_representation, axis=0)
                    print(" ###########  Time for clip (21 forward passes): ", (time.time() - start))

                    if example == 0:
                        data_set = frame_representation
                    else:
                        data_set = np.concatenate((data_set, frame_representation), axis=0)

                    print(data_set.shape)

                hkl.dump(data_set, 'final_training_data/conv_features_train-' + str(number) + '.hkl', mode='w',
                         compression='gzip', compression_opts=9)
                print("Section Saved")