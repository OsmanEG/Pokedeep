#Importing required packages
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array

#Loading in and processing the images
class LoadingInPokemon:
    def __init__(self, width, height):
        #Initializing the attributes
        self.width =  width
        self.height = height

    #Loading in and processing the data
    def dataProcessing(self):
        #Initializing lists for the features and targets
        data = []
        labels = []

        #Data location
        imageLocs = list(paths.list_images("Pokemon Gen 1 Dataset"))

        print("|POKEDEEP| Loading in images...")

        counter = 0

        #Loading in individual images
        for imageLoc in imageLocs:

            counter += 1

            #Grabbing the pokemon image and label
            image = cv2.imread(imageLoc)
            label = imageLoc.split(os.path.sep)[-2]

            #Process the image to the standardized size
            image = cv2.resize(image, (self.width, self.height), interpolation = cv2.INTER_AREA)
            image = img_to_array(image, data_format = None)

            data.append(image)
            labels.append(label)

            #Displaying image update
            if counter%1000 == 0:
                print("|POKEDEEP| Loaded in image {}/{}".format(counter, len(imageLocs)))
            
                
        #Returning a tuple of the images and class labels

        print("|POKEDEEP| Completed image loading...")

        return (np.array(data), np.array(labels))

#Initializing the image data
height = 128
width = 128
depth = 3

shape = (height, width, depth)

(data, labels) = LoadingInPokemon(width, height).dataProcessing()

#Normalizing pixel values to be between 0 and 1
data = data.astype("float") / 255.0

#Splitting the training and test data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.2, 
    random_state = 42)

#One Hot Encode the train and test labels
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)

trainY = to_categorical(trainY)
testY = to_categorical(testY)

#Creating the model
print("|POKEDEEP| Creating the network...")
model = Sequential()

#Checking data format in the keras backend
if K.image_data_format() == "channels_first":
    shape = (depth, height, width)

#Building the layers of the model

epochs = 100

print("|POKEDEEP| Compiling the network...")
model.add(Conv2D(64, kernel_size = 3, activation = "relu", input_shape = shape))
model.add(Conv2D(32, kernel_size = 3, activation = "relu"))
model.add(Flatten())
model.add(Dense(149))
model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

#Training the model
print("|POKEDEEP| Training the network...")
M = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 32, epochs = epochs)

#Saving the model
model.save("pokedeep.h5")

#Plotting loss and accuracy in training
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), M.history["loss"], label = "Training Loss")
plt.plot(np.arange(0, epochs), M.history["val_loss"], label = "Loss")
plt.plot(np.arange(0, epochs), M.history["accuracy"], label = "Training Accuracy")
plt.plot(np.arange(0, epochs), M.history["val_accuracy"], label = "Accuracy")
plt.title("Loss and Accuracy during the 'Training' process ")
plt.xlabel("Epoch Number")
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()

plt.savefig("Training Loss and Accuracy.png")


