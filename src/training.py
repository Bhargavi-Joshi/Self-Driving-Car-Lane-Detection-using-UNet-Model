import sys
import pickle
import tensorflow as tf
import numpy as np
from classical_utils import *
from unet import UNET
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.callbacks import History

def training_pipeline(batch_size=16):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=15372)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    # Loading train and test data
    train = np.array(pickle.load(open(r"full_CNN_train.p", 'rb')))
    labels = np.array(pickle.load(open(r"full_CNN_labels.p", 'rb')))
    train = train[0:6000]
    labels = labels[0:6000]
    # Reshaping and Normalizing training data
    train = np.reshape(train, (-1, 80, 160, 3)) / 255
    labels = np.reshape(labels, (-1, 80, 160, 1)) / 255
    # Generating train and test datasets from the original dataset after shuffling, splitting, and creating batches.
    train_dataset, test_dataset = prepare_dataset(train, labels, batch_size=batch_size)
    # Training
    my_unet = UNET(input_shape=train[0].shape, trainable=True, start_filters=15, name="trial unet")
    my_unet.model.summary()
    history: History = train_model(my_unet, train_dataset=train_dataset, test_dataset=test_dataset)

    # Calculate memory usage
    params_count = np.sum([np.prod(w.shape) for w in my_unet.model.trainable_weights])
    memory_bytes = params_count * 4  # Assuming float32, each parameter takes 4 bytes
    memory_megabytes = memory_bytes / (1024 ** 2)
    print(f"Memory usage after training: {memory_megabytes:.2f} MB")

# Generating a separate train dataset and test dataset depending on the train split. Shuffle, split, and create batches.
def prepare_dataset(train: list, labels: list, train_split: float = 0.8, batch_size=32):
    num_elements = len(train)
    # Loading as tf.data.Dataset
    dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(train), tf.convert_to_tensor(labels)))
    # Shuffling the dataset to increase robustness
    dataset = dataset.shuffle(num_elements)
    # Splitting the dataset into train and test based on the split
    train_size = int(train_split * num_elements)
    test_size = num_elements - train_size
    train_dataset = dataset.skip(test_size)
    test_dataset = dataset.take(test_size)
    # Creating batches for both datasets
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(1)
    return train_dataset, test_dataset


# Training the UNET model and saves a weights file and a keras model folder which can be used for predicting.
def train_model(model: UNET, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset):
    # Creating callbacks to be executed while training
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras_model_checkpoint.keras", verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=2, min_lr=0.00001, verbose=1)
    # Setting hyperparameters
    epochs = 5
    callbacks = [early_stopping, model_checkpoint, reduce_lr]
    loss = "binary_crossentropy"
    optimizer="adam"
    metrics=["accuracy"]
    model.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    # Training the model
    if model.trainable:
        history = model.model.fit(
                    train_dataset,
                    validation_data=test_dataset,
                    epochs=epochs,
                    callbacks=callbacks
                )
        # Release memory
        tf.keras.backend.clear_session()
        del model
        return history
    else:
        raise ValueError (f"smodel.trainable value is {model.trainable}. Please set the value to True in order to train the model.")

if __name__ == '__main__':
    batch_size = 16
    print(f"Using batch size: {batch_size}")
    training_pipeline(batch_size=batch_size)
