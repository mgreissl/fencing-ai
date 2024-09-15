import tensorflow as tf
import numpy as np
import argparse
import time
import subprocess as sp
import os
import hickle as hkl
from tensorflow.keras.optimizers import legacy as optimizers

num_layers = 4
drop_out_prob = 0.5
batch_size = 16
epochs = 10
learning_rate = 1e-5
test_size = 300
validation_size = 300

videos_loaded = 0
for i in os.listdir('final_training_data'):
    if i.endswith(".hkl") and 'features' in i:
        print(i)
        if videos_loaded == 0:
            loaded = hkl.load('final_training_data/' + i)
        else:
            loaded = np.concatenate((loaded, hkl.load('final_training_data/' + i)), axis=0)
        videos_loaded += 1
        print(loaded.shape)

videos_loaded = 0
for i in os.listdir('final_training_data'):
    if i.endswith(".hkl") and "labels" in i:
        print(i)
        if videos_loaded == 0:
            labels = hkl.load('final_training_data/' + i)
        else:
            labels = np.concatenate((labels, hkl.load('final_training_data/' + i)), axis=0)
        videos_loaded += 1
        print(labels.shape)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p, :, :], b[p, :]

loaded, labels = unison_shuffled_copies(loaded, labels)
print(loaded.shape, labels.shape)

test_set = loaded[:test_size]
test_labels = labels[:test_size]
validation_set = loaded[test_size:(validation_size + test_size)]
validation_labels = labels[test_size:(validation_size + test_size)]
loaded = loaded[(test_size + validation_size):]
labels = labels[(test_size + validation_size):]

print("Test Set Shape: ", test_set.shape)
print("Validation Set Shape: ", validation_set.shape)
print("Training Set Shape: ", loaded.shape)

hkl.dump(test_set, 'test_data.hkl', mode='w', compression='gzip', compression_opts=9)
hkl.dump(test_labels, 'test_lbls.hkl', mode='w', compression='gzip', compression_opts=9)

device_name = "/gpu:0"

logs_path = '/tmp/4_d-0.8'
display_step = 40

n_input = 2048
n_hidden = 1024
n_classes = 3

# tf.keras input placeholders
x = tf.keras.Input(shape=(None, n_input), dtype=tf.float32)
y = tf.keras.Input(shape=(n_classes,), dtype=tf.float32)

# Define LSTM-based RNN Model
def RNN_model(n_hidden, drop_out_prob, num_layers, n_classes):
    model = tf.keras.Sequential()
    for _ in range(num_layers - 1):
        model.add(tf.keras.layers.LSTM(n_hidden, return_sequences=True, dropout=1 - drop_out_prob))
    model.add(tf.keras.layers.LSTM(n_hidden, dropout=1 - drop_out_prob))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model

model = RNN_model(n_hidden, drop_out_prob, num_layers, n_classes)
optimizer= optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training loop variables
current_epochs = 0
step = 0

# Create TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path, histogram_freq=1)

while step < (len(labels) // batch_size):
    batch_x = loaded[step * batch_size:(step + 1) * batch_size]
    batch_y = labels[step * batch_size:(step + 1) * batch_size]

    # Train on batch
    model.train_on_batch(batch_x, batch_y)
    # Every display_step, evaluate validation accuracy and log
    if step % display_step == 0:
        val_accuracies = []
        for i in range(validation_size // batch_size):
            validation_batch_data = validation_set[i * batch_size:(i + 1) * batch_size]
            validation_batch_labels = validation_labels[i * batch_size:(i + 1) * batch_size]
            val_loss, val_acc = model.evaluate(validation_batch_data, validation_batch_labels, verbose=0)
            val_accuracies.append(val_acc)

        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies)
        print(f"Validation Accuracy: {avg_val_accuracy:.5f}")

        # Save model checkpoint
        model.save('fencing_AI_checkpoint.h5')

    step += 1
    if current_epochs < epochs:
        if step >= (len(labels) // batch_size):
            print("###################### New epoch ##########")
            current_epochs += 1
            learning_rate = learning_rate - (learning_rate * 0.15)
            step = 0

            # Shuffle the data
            loaded, labels = unison_shuffled_copies(loaded, labels)

print("Learning finished!")

# Test the model
test_accuracies = []
for i in range(test_size // batch_size):
    test_batch_data = test_set[i * batch_size:(i + 1) * batch_size]
    test_batch_labels = test_labels[i * batch_size:(i + 1) * batch_size]
    test_loss, test_acc = model.evaluate(test_batch_data, test_batch_labels, verbose=0)
    test_accuracies.append(test_acc)

avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)
print(f"Test Accuracy: {avg_test_accuracy:.5f}")
