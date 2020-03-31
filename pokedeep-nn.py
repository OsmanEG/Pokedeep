#Importing required packages
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import keras

from sklearn.preprocessing      import LabelEncoder
from sklearn.model_selection    import train_test_split
from imutils                    import paths

from keras.models               import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core          import Activation
from keras.layers.core          import Flatten
from keras.layers.core          import Dropout
from keras.layers.core          import Dense
from keras                      import backend as K
from keras.utils                import to_categorical
from keras.preprocessing.image  import img_to_array
from keras.preprocessing.image  import ImageDataGenerator

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

#Initializing image augmentation
datagen = ImageDataGenerator(width_shift_range=0.15, 
                             height_shift_range=0.15, 
                             rotation_range=1, 
                             shear_range=0.2, 
                             channel_shift_range=10, 
                             horizontal_flip=True, 
                             vertical_flip=True, 
                             zoom_range=0.1)

#Initializing the image data
height = 224
width = 224
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

#Augmenting train data
augment_iter = datagen.fit(trainX)

#Creating the model
print("|POKEDEEP| Creating the network...")
model = Sequential()

#Checking data format in the keras backend
if K.image_data_format() == "channels_first":
    shape = (depth, height, width)

#Building the layers of the model

epochs = 100

print("|POKEDEEP| Compiling the network...")
model.add(ZeroPadding2D((1,1), input_shape = shape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(149, activation = 'softmax'))

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

#Training the model
print("|POKEDEEP| Training the network...")
#M = model.fit_generator(datagen.flow(trainX, trainY, batch_size=32), steps_per_epoch=len(trainX) // 32, epochs=epochs)
M = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 16, epochs = epochs)

#Saving the model
model.save("pokedeep.h5")

#Plotting loss and accuracy in training
plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, epochs), M.history["loss"],         label = "Training Loss")
plt.plot(np.arange(0, epochs), M.history["val_loss"],     label = "Loss")
plt.plot(np.arange(0, epochs), M.history["accuracy"],     label = "Training Accuracy")
plt.plot(np.arange(0, epochs), M.history["val_accuracy"], label = "Accuracy")

plt.title("Loss and Accuracy during the 'Training' process ")
plt.xlabel("Epoch Number")
plt.ylabel("Loss and Accuracy")
plt.legend()
plt.show()

plt.savefig("Training Loss and Accuracy.png")


