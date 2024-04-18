
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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

from tensorflow.keras import layers

# Define a custom weight constraint
class WeightConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

from tensorflow.keras import backend as K
rel_max_mean=0.0475
rel_max_stddev=0.625
def my_relu_max(x):
    y=tf.keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
    slopex=y*1.00
    temp1=K.maximum(0.0, slopex)
    out11=K.minimum(1.00,temp1)
    noise = tf.random.normal(tf.shape(out11), mean=rel_max_mean, stddev=rel_max_stddev)
    scaled_noise=(noise*out11)/100
    out=out11+scaled_noise
    return out

rel_mean=-0.1778
rel_stddev=0.7873
def my_relu(x):
    slopex=x*1.00
    temp1=K.maximum(0.0, slopex)
    out22=K.minimum(1.0,temp1)
    rel_noise = tf.random.normal(tf.shape(out22), mean=rel_mean, stddev=rel_stddev)
    rel_scaled_noise=(rel_noise*out22)/100
    out=out22+rel_scaled_noise
    return out

min_value_weight=-1
max_value_weight=1
    

def deconv_concate(tensor, residual, nfilters, size=3, padding='valid', strides=(3, 3), output_padding=2):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size),kernel_constraint=WeightConstraint(min_value_weight, max_value_weight), strides=strides, padding=padding,output_padding=output_padding)(tensor)
    y = concatenate([y, residual], axis=3)
    return y    
    
def Unet(h, w, filters, num_classes = 32):
# down
    input_layer = Input(shape=(h, w, 3), name='image_input')
    
    conv1 = layers.Conv2D(filters, (3,3), kernel_initializer='he_uniform',kernel_constraint=WeightConstraint(min_value_weight, max_value_weight),strides=(1,1),padding='same')(input_layer)
    relu_max1 = my_relu_max(conv1)
    
    conv2 = layers.Conv2D(filters*2, (3,3), kernel_initializer='he_uniform',kernel_constraint=WeightConstraint(min_value_weight, max_value_weight),strides=(1,1),padding='same')(relu_max1)
    relu_max2 = my_relu_max(conv2)
    
    conv3 = layers.Conv2D(filters*4, (3,3), kernel_initializer='he_uniform',kernel_constraint=WeightConstraint(min_value_weight, max_value_weight),strides=(1,1),padding='same')(relu_max2)
    relu_max3 = my_relu_max(conv3)
    
    conv4 = layers.Conv2D(filters*8, (3,3), kernel_initializer='he_uniform',kernel_constraint=WeightConstraint(min_value_weight, max_value_weight),strides=(1,1),padding='same')(relu_max3)
    relu4 = my_relu(conv4)
    
    
# up
    deconv_concat5=deconv_concate(relu4, residual=conv3, nfilters=filters*4)#originally residual is after conv&relu but before max pool
    conv5 = layers.Conv2D(filters*4, (3,3), kernel_initializer='he_uniform',kernel_constraint=WeightConstraint(min_value_weight, max_value_weight),strides=(1,1),padding='same')(deconv_concat5)
    relu5 = my_relu(conv5)
    
    deconv_concat6=deconv_concate(relu5, residual=conv2, nfilters=filters*2)#originally residual is after conv&relu but before max pool
    conv6 = layers.Conv2D(filters*2, (3,3), kernel_initializer='he_uniform',kernel_constraint=WeightConstraint(min_value_weight, max_value_weight),strides=(1,1),padding='same')(deconv_concat6)
    relu6 = my_relu(conv6)
    
    deconv_concat7=deconv_concate(relu6, residual=conv1, nfilters=filters)#originally residual is after conv&relu but before max pool
    conv7 = layers.Conv2D(filters, (3,3), kernel_initializer='he_uniform',kernel_constraint=WeightConstraint(min_value_weight, max_value_weight),strides=(1,1),padding='same')(deconv_concat7)
    relu7 = my_relu(conv7)
    
    output_layer = Conv2D(filters=num_classes, kernel_size=(1, 1),kernel_constraint=WeightConstraint(min_value_weight, max_value_weight), activation='softmax')(relu7)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model


model = Unet(img_size , img_size , 64)
model.summary()



model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy() ,metrics=['accuracy'])

#mc = ModelCheckpoint(mode='max', filepath='top-weights-na.h5', monitor='val_accuracy',save_best_only='True', verbose=1)
#es = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0)

train_steps = train_generator.__len__()
val_steps = val_generator.__len__()

print(train_steps, val_steps)



#results = model.fit_generator(train_generator , steps_per_epoch=train_steps ,epochs=30,
                            #  validation_data=val_generator,validation_steps=val_steps,
                            # verbose = 1)#callbacks=[mc,es],


# # Custom callback to track weight changes after each batch
# class WeightTrackerCallback(tf.keras.callbacks.Callback):
#     def __init__(self, layer_name):
#         super(WeightTrackerCallback, self).__init__()
#         self.layer_name = layer_name
#         self.prev_weights = None

#     def on_batch_begin(self, batch, logs=None):
#         layer = self.model.get_layer(name=self.layer_name)
#         self.prev_weights = layer.get_weights()

#     def on_batch_end(self, batch, logs=None):
#         layer = self.model.get_layer(name=self.layer_name)
#         new_weights = layer.get_weights()
        
#         changes = [tf.reduce_sum(tf.abs(tf.subtract(prev, new)))
#                    for prev, new in zip(self.prev_weights, new_weights)]
        
#         print(f"Batch {batch + 1}, Weight changes for layer '{self.layer_name}': {changes}")



# from tensorflow.keras.callbacks import LambdaCallback
# # Callback to print weights after each batch and epoch
# class PrintWeights(LambdaCallback):
#     def __init__(self):
#         self.total_weight_change_batch = 0  # Initialize the shared variable
#         self.total_weight_change_epoch = 0
    
#     def on_batch_end(self, batch, logs=None):
#         #print("Batch {} weights:".format(batch))
#         for layer in model.layers:
#             weights = layer.get_weights()
#             if weights:
#                 current_weights = weights[0].flatten()  # Flatten to compare element-wise
#                 initial_weights = layer.initial_weights[0].flatten()  # Get initial weights
#                 weight_change = np.abs(current_weights - initial_weights).sum()  # Calculate total change
#                 self.total_weight_change_batch += weight_change

#     def on_epoch_end(self, epoch, logs=None):
#         self.total_weight_change_epoch = self.total_weight_change_batch
#         self.total_weight_change_batch= 0
#         print("Total weight change for epoch", self.total_weight_change_epoch)


#my_callback = PrintWeights()

from keras.callbacks import Callback
class WeightTracker(Callback):
    def __init__(self):
        super(WeightTracker, self).__init__()
        self.initial_weights = None  # Store initial weights
        self.final_weights = []  # Store final weights for each batch

    def on_train_begin(self, logs=None):
        self.initial_weights = self.model.get_weights()

    def on_batch_end(self, batch, logs=None):
        final_weights = self.model.get_weights()  # Get final weights after batch update
        self.final_weights.append(final_weights)
        
        
        
total_weight_diff=[]


no_epochs = 150
weight_tracker = WeightTracker()
results = model.fit_generator(train_generator , steps_per_epoch=train_steps ,epochs=no_epochs,validation_data=val_generator,validation_steps=val_steps,verbose = 1,callbacks=[weight_tracker])


#old_weights = weight_tracker.initial_weights  # Access from the instance
weight_differences = []  # Store weight differences for each batch
for i in range(len(weight_tracker.final_weights) - 1):
    current_weights = weight_tracker.final_weights[i]
    next_weights = weight_tracker.final_weights[i + 1]
    weight_diff = [np.sum(np.square(np.subtract(cw, nw))) for cw, nw in zip(current_weights, next_weights)]

    sum_weight_diff=np.sum(weight_diff)
    total_weight_diff.append(sum_weight_diff)
    #old_weights = weight_tracker.final_weights[i]

plt.plot(total_weight_diff, label='weight diff squared over all batches')
plt.show()
# Calculate the size of each part
# part_size = len(total_weight_diff) // no_epochs

# # Divide the list into 5 parts using list slicing
# parts = [total_weight_diff[i * part_size:(i + 1) * part_size] for i in range(no_epochs)]

# # Add the elements in each part
# part_sums = [sum(part) for part in parts]

# print("Sum of each part:", part_sums)

# plt.plot(part_sums, label='weight diff squared over epochs')
# plt.show()



first_part_size = 91
remaining_part_size = 92

# Calculate the number of remaining parts
remaining_parts = len(total_weight_diff) - first_part_size

# Divide the list into the first part and the remaining parts
first_part = total_weight_diff[:first_part_size]
remaining_data = total_weight_diff[first_part_size:]

# Divide the remaining data into equal parts
remaining_parts = [remaining_data[i * remaining_part_size:(i + 1) * remaining_part_size] for i in range(no_epochs-1)]

# Combine all parts into a single list
all_parts = [first_part] + remaining_parts

part_sums = [sum(part) for part in all_parts]

print("Sum of each part:", part_sums)

plt.plot(part_sums, label='weight diff squared over epochs')
plt.show()


# Print the sizes of all parts
print("Sizes of all parts:", [len(part) for part in all_parts])


# initial_weights = weight_tracker.initial_weights  # Access from the instance
# final_weights_per_batch = weight_tracker.final_weights  # Access from the instance



import pandas as pd


# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(results.history) 


# or save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)



# Create a DataFrame using existing variables
df = pd.DataFrame({
    'weight_diff_squared_batch_wise': total_weight_diff,
})


# Save each variable in a new column of the Excel sheet
df.to_excel('weight_diff_squared_batch_wise.xlsx')


# Create a DataFrame using existing variables
df = pd.DataFrame({
    'weight_diff_squared_epoch_wise': part_sums,
})


# Save each variable in a new column of the Excel sheet
df.to_excel('weight_diff_squared_epoch_wise.xlsx')


#%%

#model.save('camvid_dataset/camvid_unet_model_na_kaggle_trained.h5')

#trained_model = keras.models.load_model("camvid_dataset/camvid_unet_model_na_kaggle_trained.h5")
trained_model=model
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
    #plt.title(title)
    for i in range(4):
        #print(pred[i].shape)
        plt.subplot(2, 4, i+1)
        #plt.imshow(img[i].astype('uint8'))
        plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
        plt.imshow(img[i])
    plt.show()
    
    
pred = np.argmax(y_pred, axis=3)
y_pred_rgb = map_this(pred,class_map)
#test = np.argmax(y_test, axis=3)
y_test_rgb = map_this(y_test,class_map)

plot_result(x_test,"Test Images")

plot_result(y_test_rgb,"Original Masks")

plot_result(y_pred_rgb,"Predicted mask")
