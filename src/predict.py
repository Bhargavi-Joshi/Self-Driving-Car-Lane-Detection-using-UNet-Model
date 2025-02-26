import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
from keras import Model

def predict_lanes():
    # Loading train and test data
    train = np.array(pickle.load(open(r"full_CNN_train.p", 'rb')))
    labels = np.array(pickle.load(open(r"full_CNN_labels.p", 'rb')))
    # Finding image number as given through cli from train data
    image = np.array(train[int(sys.argv[1])]) / 255
    ground_truth = np.array(labels[int(sys.argv[1])]) / 255
    # Loading saved model
    my_unet: Model = tf.keras.models.load_model('keras_model_checkpoint.keras')
    # Generating predictions
    image_reshaped = np.reshape(image, [1, 80, 160, 3])
    prediction = my_unet.predict(image_reshaped)
    prediction = np.reshape(prediction, (80, 160, 1))
    # Binarizing values to calculate IoU
    prediction = prediction.round()
    ground_truth = ground_truth.round()
    # Calculating IoU
    iou = calculate_iou(ground_truth, prediction)
    images = [
        (f"Train Example, image no. {sys.argv[1]}", image),
        (f"Ground Truth, image no. {sys.argv[1]}", ground_truth),
        (f"Predicted Output, IoU: {iou:.2f}", prediction)
    ]
    # Plotting results
    fig = plt.figure()
    for i, image_data in list(zip(range(1,len(images) + 1), images)):
        title, image = image_data
        fig.add_subplot(1, len(images), i)
        plt.title(title)
        plt.axis("off")
        plt.imshow(image)
    plt.show()

# Calculating Intersection over Union (IoU)
def calculate_iou(y_true, y_pred):
    # y_true = ground truth image
    # y_pred = output predicted by model
    # iou = Always between 0 and 1
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        for j in range(len(y_true[0])):
            if y_true[i][j] == 1 and y_pred[i][j] == 1:
                tp += 1
            elif y_true[i][j] == 0 and y_pred[i][j] == 1:
                fp += 1
            elif y_true[i][j] == 1 and y_pred[i][j] == 0:
                fn += 1
    # Calculate IoU
    iou = tp / (tp + fp + fn)
    return iou
 

if __name__ == '__main__':
    predict_lanes()