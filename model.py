import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    #new_img = img[35:140,:,:]
    # crop to 40x320x3
    new_img = img[50:140,:,:]
    #plt.imshow(new_img)
    #plt.show()
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    #plt.imshow(new_img)
    #plt.show()
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    #plt.imshow(new_img)
    #plt.show()
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    #plt.imshow(new_img)
    #plt.show()
    return new_img

def bright_augment(img):
    img1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    #print(random_bright)
    #print(img1[:,:,2])
    img1[:,:,2] = img1[:,:,2]*random_bright
    img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2RGB)
    return img1

def data_load():
    img_dir = "./data/IMG/"
    csv_dir = './data/driving_log.csv'
    lines = []
    images = []
    steering_angles = []
    channels = 3
    rows = 64
    columns = 64 #Resized images in rows and columns
    correction = 0.2 #Correction factor when considering left and right camera images for adjustment


    ##Reading training data from csv file into lines
    with open(csv_dir) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        del lines[0]

    print("Number of centre images in original dataset are {}".format(len(lines)))

    for line in lines:
        for i in range(3):
            source_path = line[i]
            #source_path = line[0]
            tokens = source_path.split('/')
            image_name = tokens[-1]
            image_path = img_dir + image_name
            image = cv2.imread(image_path)
            plt.imshow(image)
            plt.show()
            #image = mpimg.imread(image_path)
            ##Resize Original size (160,320,3) to (66,200,3) based on Nvidia's Model
            image = preprocess_image(image)
            #plt.imshow(image)
            #plt.show()
            #exit()
            image = bright_augment(image)
            #plt.imshow(image)
            #plt.show()
            images.append(image)

        ##Reading and appending measurement with necessary correction factor to measurement list
        steering_angle = float(line[3])
        steering_angles.append(steering_angle)
        steering_angles.append(steering_angle + correction)
        steering_angles.append(steering_angle - correction)

    #To ensure length of images equals steering angles
    if (len(images) == len(steering_angles)):
        print("Length of images, steering angles list are equal to {}".format(len(images)))
    else:
        print("Images and measurements list is not equal")
        exit()

    return images, steering_angles

def flipped_augmentation(image, measurements):
    augmented_images = []
    augmented_measurements = []

    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = float(measurement) * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)

    print("Number of augmented images generated is {}".format(len(augmented_images))) #

    X_train = np.asarray(augmented_images)
    y_train = np.asarray(augmented_measurements)

    return X_train, y_train

#Create Model
def autonomous_model():
    lr = 0.0001
    weight_init='glorot_normal'
    loss = 'mean_squared_error'

    model = Sequential()

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv1'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv2'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), activation='relu', name='Conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    #model.add(BatchNormalization())
    model.add(Convolution2D(128, 2, 2, border_mode='same', subsample=(1, 1), activation='relu', name='Conv4'))
    #model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='FC2'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='FC3'))
    model.add(Dense(1))

    return model


#Load the data
images, measurements = data_load()
print("Data Loading Complete")
print("The steering angles ranges from {} to {}".format(min(measurements), max(measurements)))
#exit()

#Apply flipped image augmentation
features, labels = flipped_augmentation(images, measurements)
print("Flipped augmentation complete")
print(features.shape)
#exit()

#Compile Generation
model = autonomous_model()
model.summary()
print("Model Generated")
#exit()

# Adding Callback functions
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

#Model Compilation
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer = 'adam', loss = 'mse')
print("Model Compiled")
#exit()
model.fit(features, labels, batch_size=32, validation_split=0.2, shuffle = True, nb_epoch = 10, callbacks=[save_weights, early_stopping])
print("Model Fit")
#exit()

#Save Model
model.save('model.h5')
print("Model Saved.Phew! Job Done")
