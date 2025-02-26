from keras.layers import Conv2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.layers import concatenate, Input
from keras import Model

class UNET():
    # Initializing a UNET model and returns a model object that can be trained.
    def __init__(self, input_shape: tuple[int], trainable: bool, start_filters=16, name = 'default_unet'):
        self.name = name
        # input_shape - The input shape that the Input() layer of the model will accept.
        self.input_shape = input_shape
        # start_filters - Set the value of the starting number of filters for the model. They will increase to 8x the starting number at the deepest layers.
        self.start_filters = start_filters
        # trainable - Boolean value to signify if the model is trainable
        self.trainable = trainable
        self.input_layer = Input(self.input_shape)
        self.output_layer = self.build_unet(input_layer=self.input_layer, starting_filters=self.start_filters)
        self.model = Model(self.input_layer, self.output_layer)

    # Initialises the UNET Model and returns the output layer.
    def build_unet(self, input_layer, starting_filters, kernel_size = (3,3)):
        # The first input layer is passed as an argument
        # Contraction Path Starts 
        # Convolutional Block 1 with 2 layers of Convolution (3x3), Max Pooling Layer and a Dropout Layer
        conv1 = Conv2D(starting_filters * 1, kernel_size, activation="relu", padding="same")(input_layer)
        conv1 = Conv2D(starting_filters * 1, kernel_size, activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)
        # Convolutional Block 2 with 2 layers of Convolution (3x3), Max Pooling Layer and a Dropout Layer
        conv2 = Conv2D(starting_filters * 2, kernel_size, activation="relu", padding="same")(pool1)
        conv2 = Conv2D(starting_filters * 2, kernel_size, activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)
        # Convolutional Block 3 with 2 layers of Convolution (3x3), Max Pooling Layer and a Dropout Layer
        conv3 = Conv2D(starting_filters * 4, kernel_size, activation="relu", padding="same")(pool2)
        conv3 = Conv2D(starting_filters * 4, kernel_size, activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)
        # Convolutional Block 4 with 2 layers of Convolution (3x3), Max Pooling Layer and a Dropout Layer
        conv4 = Conv2D(starting_filters * 8, kernel_size, activation="relu", padding="same")(pool3)
        conv4 = Conv2D(starting_filters * 8, kernel_size, activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)
        # Middle Convolutional Block with 2 layers of Convolution (3x3) 
        convm = Conv2D(starting_filters * 16, kernel_size, activation="relu", padding="same")(pool4)
        convm = Conv2D(starting_filters * 16, kernel_size, activation="relu", padding="same")(convm)
        
        # Expansion Path Starts 
        # Upconvolutional Block 4 with Up Convolution, Skip connection (4), Dropout layer, and 2 Convolutional Layers
        deconv4 = Conv2DTranspose(starting_filters * 8, kernel_size, strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4]) # Skip connection from 4th convolutional block
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(starting_filters * 8, kernel_size, activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(starting_filters * 8, kernel_size, activation="relu", padding="same")(uconv4)
         # Upconvolutional Block 3 with Up Convolution, Skip connection (3), Dropout layer, and 2 Convolutional Layers
        deconv3 = Conv2DTranspose(starting_filters * 4, kernel_size, strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3]) # Skip connection from 3rd convolutional block
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(starting_filters * 4, kernel_size, activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(starting_filters * 4, kernel_size, activation="relu", padding="same")(uconv3)
        # Upconvolutional Block 2 with Up Convolution, Skip connection (2), Dropout layer, and 2 Convolutional Layers
        deconv2 = Conv2DTranspose(starting_filters * 2, kernel_size, strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2]) # Skip connection from 2nd convolutional block
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(starting_filters * 2, kernel_size, activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(starting_filters * 2, kernel_size, activation="relu", padding="same")(uconv2)
        # Upconvolutional Block 1 with Up Convolution, Skip connection (1), Dropout layer, and 2 Convolutional Layers
        deconv1 = Conv2DTranspose(starting_filters * 1, kernel_size, strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1]) # Skip connection from 1st convolutional block
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(starting_filters * 1, kernel_size, activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(starting_filters * 1, kernel_size, activation="relu", padding="same")(uconv1)
        # Final Output layer with Sigmoid activation to provide us a segmentation map
        output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
        # A layer with sigmoid activation to provide us our segmentation prediction.
        return output_layer
