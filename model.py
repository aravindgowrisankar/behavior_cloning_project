import csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from keras.layers import Flatten,Dense,Convolution2D,MaxPooling2D,Dropout,Activation,Lambda,Cropping2D
from keras.models import Sequential

#two or three laps of center lane driving
#one lap of recovery driving from the sides
#one lap focusing on driving smoothly around curves

import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

correction=[0,0.2,-0.2]#fixed steering corrections. trig formula could be applied
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #Data Augmentation 1: Using All 3 images center, left and right
                for i in range(3):
                    file_path=batch_sample[i]
                    #Data Augmentation 2: Flipping Image Laterally
                    original_image = np.array(Image.open(file_path))
                    flipped_image = np.fliplr(original_image)
                    angle = float(batch_sample[3])
                    adjusted_angle=angle+correction[i]
                    images.append(original_image)
                    angles.append(adjusted_angle)
                    images.append(flipped_image)
                    angles.append(-adjusted_angle)                    
# trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # print("X_train.shape",X_train.shape)
            # print("y_train.shape",y_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)
def load_data():
    """ Load images including provided data and self-generated data."""
    directories=["./track1/",
                 "./track1_recovery/",
                 "./track2/",
                 "./track1_reverse/",
                 "./track2_reverse/",#Additional data for model built on top of lenet.h5
                 "./track2_recovery/",#Additions data for model built on top of lenet.h5
             ]
    lines=[]
    for directory in directories:
        with open(directory+"driving_log.csv") as csvfile:
            reader=csv.reader(csvfile)
            for line in reader:
                lines.append(line)
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    return train_samples, validation_samples


def LeNet():
    """LeNet architecture(final solution)"""
    model=Sequential()
    model.add(Lambda(lambda x: (x-128.0)/128.0,input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    # SOLUTION: Layer 1: Convolutional. Input =  Output = 
    model.add(Convolution2D(nb_filter=6, nb_row=5,nb_col=5, subsample=(1,1),bias=True))

    # SOLUTION: Activation.
    model.add(Activation('relu'))


    # SOLUTION: Pooling. Input =  Output = 
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode="valid"))

    # SOLUTION: Layer 2: Convolutional. Output = 
    model.add(Convolution2D(nb_filter=16, nb_row=5,nb_col=5, subsample=(1,1),bias=True))
    
    # SOLUTION: Activation.
    model.add(Activation('relu'))

    # SOLUTION: Pooling. Input =  Output = 
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode="valid"))

    # SOLUTION: Flatten. Input = 5x5x16. Output = .
    model.add(Flatten())
    
    # SOLUTION: Layer 3: Fully Connected. Input = . Output = .
    model.add(Dense(120,bias=True))
    
    # SOLUTION: Activation.
    model.add(Activation('relu'))

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    model.add(Dense(84,bias=True))
    
    # SOLUTION: Activation.
    model.add(Activation('relu'))

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 1.
    model.add(Dense(1))    
    model.compile(loss="mse",optimizer="adam")
    return model

def first_model():
    """ First Network I tried based on code from the Keras chapter."""
    model=Sequential()
    #    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x-128.0)/128.0,input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(32, 3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss="mse",optimizer="adam")
    return model

def plot(history_object):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("loss_with_generator.png")


def train_model(model,train_samples,validation_samples,batch_size):
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    history=model.fit_generator(train_generator, 
                                samples_per_epoch= len(train_samples)*6, 
                                validation_data=validation_generator,
                                nb_val_samples=len(validation_samples)*6, nb_epoch=3)
    return history


if __name__=="__main__":
    old_model_name="lenet"
    model_name="model"
    model=LeNet()
    if old_model_name:
        model.load_weights(old_model_name+"_weights.h5")
    train_samples,validation_samples=load_data()
    print("train_samples",len(train_samples),train_samples[0])
    print("validation_samples",len(validation_samples),validation_samples[0])
    history=train_model(model,train_samples,validation_samples,batch_size=64)
    model.save(model_name+".h5")
    model.save_weights(model_name+"_weights.h5")
    plot(history)
