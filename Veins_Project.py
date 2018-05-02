
# coding: utf-8

# In[8]:


import os
import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from PIL import Image        
from skimage.io import imread
from skimage.transform import resize


# In[3]:


import Augmentor
Augmentor.__version__


# In[4]:


from keras.models import load_model
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


# In[5]:


import sys
sys.path


# ## Load and resize the images

# In[6]:


IMG_HEIGHT, IMG_WIDTH = 128, 256
SEED=42

IMAGE_LIB = 'C:\\Users\\kovsa\\input\\new_images\\'
MASK_LIB = 'C:\\Users\\kovsa\\input\\new_masks\\'


# In[10]:


all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.bmp']

x_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(os.path.join(IMAGE_LIB, name), cv2.IMREAD_UNCHANGED).astype("int16").astype('float32')
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    x_data[i] = im

y_data = np.empty((len(all_images), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
for i, name in enumerate(all_images):
    im = cv2.imread(os.path.join(MASK_LIB, name), cv2.IMREAD_GRAYSCALE).astype('float32')/255.
    im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    y_data[i] = im


# In[11]:


fig, ax = plt.subplots(1,2, figsize = (8,4))
ax[0].imshow(x_data[0], cmap='gray')
ax[1].imshow(y_data[0], cmap='gray')
plt.show()


# In[12]:


x_data = x_data[:,:,:,np.newaxis]
y_data = y_data[:,:,:,np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.5)


# ## Define and train model

# In[13]:


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


# In[14]:


input_layer = Input(shape=x_train.shape[1:])
c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
l = MaxPool2D(strides=(2,2))(c1)
c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c2)
c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
l = MaxPool2D(strides=(2,2))(c3)
c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
l = Dropout(0.5)(l)
output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
                                                         
model = Model(input_layer, output_layer)


# In[15]:


model.summary()


# In[14]:


# p = Augmentor.Pipeline("C://Users/kovsa/input/images")
# p.ground_truth("C://Users/kovsa/input/masks")

# p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.8)
# p.flip_top_bottom(probability=0.5)

# p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.8)
# p.flip_top_bottom(probability=0.5)
# #p.random_erasing(0.4,0.4)
# # p.add_operation(RandomErasing(0.4,0.4))

# p.sample(10)


# In[15]:


# p = Augmentor.Pipeline("C://Users/kovsa/input/images")
# # Point to a directory containing ground truth data.
# # Images with the same file names will be added as ground truth data
# # and augmented in parallel to the original data.
# p.ground_truth("C://Users/kovsa/input/masks")
# # Add operations to the pipeline as normal:
# p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
# p.flip_left_right(probability=0.5)
# p.zoom_random(probability=0.5, percentage_area=0.8)
# p.flip_top_bottom(probability=0.5)
# p.random_distortion(0.5,16,16,4)
# p.skew(0.75, 0.25)
# p.sample(10000)


# In[18]:


# g = p.keras_generator(batch_size= 8)
# # g = p.keras_generator_from_array(x_train, y_train, batch_size=8)
# image_batch, mask_batch = next(g)
# fix, ax = plt.subplots(8,2, figsize=(8,20))
# for i in range(8):
#     ax[i,0].imshow(image_batch[i,:,:,0], cmap='gray')  
#     ax[i,1].imshow(mask_batch[i,:,:,0], cmap='gray')
# plt.show()


# In[20]:


# mask_batch.shape

************************************************************************************
# In[21]:


def my_generator(x_train, y_train, batch_size):    
    data_generator = ImageDataGenerator()
    data_generator.fit(x_train, batch_size, seed=SEED)
    data_generator = data_generator.flow(x_train, batch_size=batch_size, seed=SEED)
    
    mask_generator = ImageDataGenerator()
    mask_generator.fit(y_train, batch_size, seed=SEED)
    mask_generator = mask_generator.flow(y_train, batch_size=batch_size, seed=SEED)

    return zip(data_generator, mask_generator)


# By using the same RNG seed in both calls to ImageDataGenerator, we should get images and masks that correspond to each other. Let's check this, to be safe.

# In[22]:


gen = my_generator(x_train, y_train, 8)
image_batch, mask_batch = next(gen)
fix, ax = plt.subplots(8,2, figsize=(8,20))
for i in range(8):
    ax[i,0].imshow(image_batch[i,:,:,0], cmap='gray')
    ax[i,1].imshow(mask_batch[i,:,:,0], cmap='gray')
plt.show()


# In[23]:


model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])


# In[24]:


from keras.callbacks import TensorBoard

weight_saver = ModelCheckpoint('checkpoints/lung_{epoch:02d}-{val_dice_coef:.2f}.h5', monitor='val_dice_coef', 
                                              save_best_only=False, verbose=1)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)
name='example'
tensor_board = TensorBoard(log_dir=os.path.join('output', name))


# In[30]:


EPOCHES = 50

hist = model.fit_generator(my_generator(x_train, y_train, 8),
                           steps_per_epoch = 20,
                           validation_data = (x_val, y_val),
                           epochs=EPOCHES,
                           callbacks = [weight_saver, annealer, tensor_board])


# In[31]:


#!ls checkpoints


# In[33]:


t_model = load_model('checkpoints/lung_50-0.95.h5', {'dice_coef': dice_coef})


# In[34]:


plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['dice_coef'], color='b')
plt.plot(hist.history['val_dice_coef'], color='r')
plt.show()


# In[35]:


plt.imshow(model.predict(x_train[0].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')


# In[37]:


y_hat = model.predict(x_val)
fig, ax = plt.subplots(1,3,figsize=(15,10))
ax[0].imshow(x_val[0,:,:,0], cmap='gray')
ax[1].imshow(y_val[0,:,:,0], cmap='gray')
ax[2].imshow(y_hat[0,:,:,0], cmap='gray')


# In[66]:


fig, ax = plt.subplots(1,3,figsize=(15,10))
ax[0].imshow(x_val[5,:,:,0], cmap='gray')
ax[1].imshow(y_val[5,:,:,0], cmap='gray')
ax[2].imshow(y_hat[5,:,:,0], cmap='gray')


# In[64]:


fig, ax = plt.subplots(1,3,figsize=(15,10))
ax[0].imshow(x_val[2,:,:,0], cmap='gray')
ax[1].imshow(y_val[2,:,:,0], cmap='gray')
ax[2].imshow(y_hat[2,:,:,0], cmap='gray')


# In[65]:


fig, ax = plt.subplots(1,3,figsize=(15,10))
ax[0].imshow(x_val[3,:,:,0], cmap='gray')
ax[1].imshow(y_val[3,:,:,0], cmap='gray')
ax[2].imshow(y_hat[3,:,:,0], cmap='gray')


# A good result, but it probably helped that these images are very homogeneous.

# In[40]:


plt.imshow(model.predict(x_train[5].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')


# In[41]:


plt.imshow(model.predict(x_train[11].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')


# In[42]:


plt.imshow(model.predict(x_train[16].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')


# In[43]:


plt.imshow(model.predict(x_train[21].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')


