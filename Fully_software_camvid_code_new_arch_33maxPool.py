
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


##### https://www.kaggle.com/code/mayankrajpurohit/camvid-segmentation-u-net/notebook

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('camvid_dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.utils import Sequence, to_categorical, plot_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Input, MaxPooling2D, concatenate, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from random import sample, choice
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


train_img_lst = os.listdir("camvid_dataset/train")
val_img_lst = os.listdir("camvid_dataset/val")
test_img_lst = os.listdir("camvid_dataset/test")
print(len(train_img_lst),len(val_img_lst), len(test_img_lst))
print(type(train_img_lst[0].split('.')[0]))


'''This function makes pairs of directories of Image and its mask '''
def make_pair(img_lst,image_dir,mask_dir):
    pairs = []
    #print(image_dir+img_lst[0])
    for im in img_lst:
        pairs.append((image_dir + im, mask_dir + im.split('.')[0]+'_L.png'))
        
    return pairs
    

'''Here we create lists of pairs of images and corresponding masks for both train and validation Images'''
train_pairs = make_pair(train_img_lst, "camvid_dataset/train/", 
                        "camvid_dataset/train_labels/")

val_pairs = make_pair(val_img_lst, "camvid_dataset/val/", 
                        "camvid_dataset/val_labels/")

test_pairs = make_pair(test_img_lst, "camvid_dataset/test/", 
                        "camvid_dataset/test_labels/")

test_pairs[0]


'''We can simply plot and see the image and corresponding mask from above list of directories randomly'''
temp = choice(train_pairs)
img = img_to_array(load_img(temp[0]))
mask = img_to_array(load_img(temp[1]))
#mask_pil = np.asarray(Image.open(temp[1]))

plt.figure(figsize=(12,12))
plt.subplot(121)
plt.title("Image")
plt.imshow(img/255)
plt.subplot(122)
plt.title("Mask")
plt.imshow(mask/255)
#plt.subplot(123)
#plt.imshow(mask_pil)
plt.show()


class_map_df = pd.read_csv("camvid_dataset/class_dict.csv")
class_map_df.head()


class_map = []
for index,item in class_map_df.iterrows():
    class_map.append(np.array([item['r'], item['g'], item['b']]))
    
print(len(class_map))
print(class_map[0])


"""This function will be used later, to assert that mask should contains values that are class labels only.
   Like, our example has 32 classes , so predicted mask must contains values between 0 to 31. 
   So that it can be mapped to corresponding RGB."""
def assert_map_range(mask,class_map):
    mask = mask.astype("uint8")
    for j in range(img_size):
        for k in range(img_size):
            assert mask[j][k] in class_map , tuple(mask[j][k])
            

'''This method will convert mask labels(to be trained) from RGB to a 2D image whic holds class labels of the pixels.'''
def form_2D_label(mask,class_map):
    mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2],dtype= np.uint8)
    
    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i
    
    return label


lab = form_2D_label(mask,class_map)
np.unique(lab,return_counts=True)


class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, pair,class_map,  batch_size=16, dim=(224,224,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.pair = pair
        self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
            # Store sample
            img = load_img(self.pair[i][0] ,target_size=self.dim)
            img = img_to_array(img)/255.
            batch_imgs.append(img)

            label = load_img(self.pair[i][1],target_size=self.dim)
            label = img_to_array(label)
            #------ comment these two lines to see proper working of datagenerator in cell below----#
            label = form_2D_label(label,self.class_map)
            #label = np.asarray(to_categorical(label , num_classes = 32))
            #------ But after that uncomment again them before training the model----------#
            #------ comment them to just run the cell below and again uncomment these two lines----#
            #print(label.shape)
            batch_labels.append(label)
        return np.array(batch_imgs) ,np.array(batch_labels)
    
    

img_size = 512
#class_map = class_palette()

train_generator1 = DataGenerator(train_pairs,class_map,batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)
X,y = train_generator1.__getitem__(0)
print(X.shape, y.shape)


plt.figure(figsize=(12, 6))
print("Images")
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(X[i])
plt.show()

print("Masks")
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.imshow(y[i])
plt.show()

####Go and uncomment those lines in Datagenerator class, __data_generation() method and run that cell again.

train_generator = DataGenerator(train_pairs,class_map,batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)
val_generator = DataGenerator(val_pairs, class_map, batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)
test_generator = DataGenerator(test_pairs, class_map, batch_size=4, dim=(img_size,img_size,3) ,shuffle=True)


def relu_max_block(tensor, size=3, padding='same', initializer="he_normal"):
    x = Activation("relu")(tensor)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    return x
    
def conv_relu_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = Activation("relu")(x)
    return x


def deconv_concate(tensor, residual, nfilters, size=3, padding='valid', strides=(3, 3), output_padding=2):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding,output_padding=output_padding)(tensor)
    y = concatenate([y, residual], axis=3)
    return y
    
from tensorflow.keras import layers
def Unet(h, w, filters, num_classes = 32):
# down
    input_layer = Input(shape=(h, w, 3), name='image_input')
    
    conv1 = layers.Conv2D(filters, (3,3), kernel_initializer='he_uniform',strides=(1,1),padding='same')(input_layer)
    relu_max1 = relu_max_block(conv1)
    
    conv2 = layers.Conv2D(filters*2, (3,3), kernel_initializer='he_uniform',strides=(1,1),padding='same')(relu_max1)
    relu_max2 = relu_max_block(conv2)
    
    conv3 = layers.Conv2D(filters*4, (3,3), kernel_initializer='he_uniform',strides=(1,1),padding='same')(relu_max2)
    relu_max3 = relu_max_block(conv3)
    
    conv4 = layers.Conv2D(filters*8, (3,3), kernel_initializer='he_uniform',strides=(1,1),padding='same')(relu_max3)
    relu4 = Activation("relu")(conv4)
    
    
# up
    deconv_concat5=deconv_concate(relu4, residual=conv3, nfilters=filters*4)#originally residual is after conv&relu but before max pool
    conv5 = layers.Conv2D(filters*4, (3,3), kernel_initializer='he_uniform',strides=(1,1),padding='same')(deconv_concat5)
    relu5 = Activation("relu")(conv5)
    
    deconv_concat6=deconv_concate(relu5, residual=conv2, nfilters=filters*2)#originally residual is after conv&relu but before max pool
    conv6 = layers.Conv2D(filters*2, (3,3), kernel_initializer='he_uniform',strides=(1,1),padding='same')(deconv_concat6)
    relu6 = Activation("relu")(conv6)
    
    deconv_concat7=deconv_concate(relu6, residual=conv1, nfilters=filters)#originally residual is after conv&relu but before max pool
    conv7 = layers.Conv2D(filters, (3,3), kernel_initializer='he_uniform',strides=(1,1),padding='same')(deconv_concat7)
    relu7 = Activation("relu")(conv7)
    
    output_layer = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax')(relu7)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model


model = Unet(img_size , img_size , 64)
model.summary()



model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy() ,metrics=['accuracy'])

mc = ModelCheckpoint(mode='max', filepath='top-weights-na.h5', monitor='val_accuracy',save_best_only='True', verbose=1)
es = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0)

train_steps = train_generator.__len__()
val_steps = val_generator.__len__()

print(train_steps, val_steps)



results = model.fit_generator(train_generator , steps_per_epoch=train_steps ,epochs=150,
                              validation_data=val_generator,validation_steps=val_steps,
                             verbose = 1)#%,callbacks=[mc,es],

model.save('camvid_dataset/camvid_unet_model_na_kaggle_trained.h5')


trained_model = keras.models.load_model("camvid_dataset/camvid_unet_model_na_kaggle_trained.h5")
trained_model.evaluate_generator(test_generator, verbose=1)


x_test, y_test = test_generator.__getitem__(2)
print(x_test.shape, y_test.shape)

y_pred = trained_model.predict(x_test, verbose = 1, batch_size = 4)
y_pred.shape


'''This converts predicted map to RGB labels'''
def map_this(y_pred,class_map):
    y_pred_rgb = np.zeros((y_pred.shape[0],y_pred.shape[1],y_pred.shape[2],3))
    for i in range(y_pred.shape[0]):
        image = np.zeros((y_pred.shape[1],y_pred.shape[2],3))
        for j in range(y_pred.shape[1]):
            for k in range(y_pred.shape[2]):
                image[j,k,:] = class_map[y_pred[i][j][k]]
        y_pred_rgb[i] = image
    return y_pred_rgb

"""This will plot original image, original mask and predicted mask"""
def plot_result(img , title):
    plt.figure(figsize=(12, 6))
    plt.title(title)
    for i in range(4):
        #print(pred[i].shape)
        plt.subplot(2, 4, i+1)
        plt.imshow(img[i])
    plt.show()
    
    
pred = np.argmax(y_pred, axis=3)
y_pred_rgb = map_this(pred,class_map)
#test = np.argmax(y_test, axis=3)
y_test_rgb = map_this(y_test,class_map)

plot_result(x_test,"Test Images")

plot_result(y_test_rgb,"Original Masks")

plot_result(y_pred_rgb,"Predicted mask")
