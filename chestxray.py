
class pkg:
  #### DOWNLOADING AND LOADING DATA
  def get_metadata(metadata_path, which_splits = ['train', 'test']):  
    
    metadata = pd.read_csv(metadata_path)
    keep_idx = metadata['split'].isin(which_splits)
    return metadata[keep_idx]

  def get_data_split(split_name, flatten, all_data, metadata, image_shape):
    
    sub_df = metadata[metadata['split'].isin([split_name])]
    index  = sub_df['index'].values
    labels = sub_df['class'].values
    data = all_data[index,:]
    if flatten:
      data = data.reshape([-1, np.product(image_shape)])
    return data, labels

  def get_train_data(flatten, all_data, metadata, image_shape):
    return get_data_split('train', flatten, all_data, metadata, image_shape)

  def get_test_data(flatten, all_data, metadata, image_shape):
    return get_data_split('test', flatten, all_data, metadata, image_shape)

  def get_field_data(flatten, all_data, metadata, image_shape):
    return get_data_split('field', flatten, all_data, metadata, image_shape)
 
import gdown
import zipfile
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from collections import Counter
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape,Dense, Conv2D, GlobalAveragePooling2D
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16
from imgaug import augmenters 

# file variables
image_data_url = 'https://drive.google.com/uc?id=1DNEiLAWguswhiLXGyVKsgHIRm1xZggt_'
metadata_url = 'https://drive.google.com/uc?id=1MW3_FU6qc0qT_uG4bzxhtEHy4Jd6dCWb'
image_data_path = './image_data.npy'
metadata_path = './metadata.csv'


gdown.download(image_data_url, './image_data.npy', True)
gdown.download(metadata_url, './metadata.csv', True)


_all_data = np.load('image_data.npy')
_metadata = pkg.get_metadata(metadata_path, ['train','test','field'])
_image_shape = [64,64,3]

get_data_split = pkg.get_data_split
get_metadata    = lambda : pkg.get_metadata(metadata_path, ['train','test'])
get_train_data  = lambda flatten = False : pkg.get_train_data(flatten = flatten, all_data =_all_data, metadata = _metadata , image_shape = _image_shape)
get_test_data   = lambda flatten = False : pkg.get_test_data(flatten = flatten, all_data = _all_data, metadata = _metadata , image_shape = _image_shape )



metadata = get_metadata()

metadata.head()


print(metadata.groupby(["class","split"]).count())

sns.countplot(x ="class", data=metadata, hue = "split")


def plot_one_image(data, labels = [], index = None, image_shape = [64,64,3]):
   
    num_dims   = len(data.shape)
    num_labels = len(labels)

    # reshape data if necessary
    if num_dims == 1:
      data = data.reshape()
    if num_dims == 2:
      data = data.reshape(np.vstack[-1, image_shape])
    num_dims   = len(data.shape)

    # check if single or multiple images
    if num_dims == 3:
      if num_labels > 1:
        print('Multiple labels does not make sense for single image.')
        return

      label = labels      
      if num_labels == 0:
        label = ''
      image = data

    if num_dims == 4:
      image = data[index, :]
      label = labels[index]

    # plot image of interest
    print('Label: %s'%label)
    plt.imshow(image)
    plt.show()

train_data, train_labels = get_train_data()
test_data, test_labels = get_test_data()

plot_one_image(train_data,train_labels,1)
plot_one_image(train_data,train_labels,40)


## Here we load the train and test data for you to use.
(train_data, train_labels) = get_train_data(flatten = True)
(test_data, test_labels) = get_test_data(flatten = True)
############

knn  = KNeighborsClassifier(n_neighbors=10)
log_reg  = LogisticRegression()
dt = DecisionTreeClassifier()

knn.fit(train_data, train_labels)
log_reg.fit(train_data, train_labels)
dt.fit(train_data, train_labels)

predictions = knn.predict(test_data)
score = accuracy_score(test_labels, predictions)
print("KNN accuracy:",score)

predictions = log_reg.predict(test_data)
score = accuracy_score(test_labels, predictions)
print("Logistic Regression accuracy:",score)

predictions = dt.predict(test_data)
score = accuracy_score(test_labels, predictions)
print("Decision Tree Classifier accuracy:",score)


def plot_acc(history, ax = None, xlabel = 'Epoch #'):
    history = history.history
    history.update({'epoch':list(range(len(history['val_accuracy'])))})
    history = pd.DataFrame.from_dict(history)
    best_epoch = history.sort_values(by = 'val_accuracy', \
                                     ascending = False).iloc[0]['epoch']
    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
    sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
    ax.axhline(0.5, linestyle = '--',color='red', label = 'Chance')
    ax.axvline(x = best_epoch, linestyle = '--', color = 'green', 
               label = 'Best Epoch')  
    ax.legend(loc = 1)    
    ax.set_ylim([0.4, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    plt.show()
    print("The highest validation accuracy was",history.sort_values
          (by = 'val_accuracy', ascending = False).iloc[0]['val_accuracy'])
    print("The lowest validation accuracy was",history.sort_values
          (by = 'val_accuracy', ascending = True).iloc[0]['val_accuracy'])
    
    
def plot_loss(history, ax = None, xlabel = 'Epoch #'):
    history = history.history
    history.update({'epoch':list(range(len(history['val_loss'])))})
    history = pd.DataFrame.from_dict(history)
    best_epoch = history.sort_values(by = 'val_loss',
                                     ascending = True).iloc[0]['epoch']
    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = 'val_loss', data = history,\
                 label = 'Validation', ax = ax)
    sns.lineplot(x = 'epoch', y = 'loss', data = history,\
                 label = 'Training', ax = ax)
    ax.axvline(x = best_epoch, linestyle = '--', color = 'green',\
               label = 'Best Epoch')  
    ax.legend(loc = 1)    
    ax.set_ylim([0.1, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Loss (Fraction)')
    plt.show()
    print("The lowest validation loss was",history.sort_values\
          (by = 'val_loss', ascending = True).iloc[0]['val_loss'])
    print("The highest validation loss was",history.sort_values\
          (by = 'val_loss', ascending = False).iloc[0]['val_loss'])

model = Sequential()
model.add(Reshape((64,64,3)))
model.add(Conv2D(32, (3, 3), padding = 'same', \
                 activation="relu",input_shape = (2000,64,64,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), padding = 'same',activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))


opt = tensorflow.keras.optimizers.RMSprop(lr=1e-4, decay=1e-6)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=['accuracy']) 

monitor = ModelCheckpoint('./model.h5', monitor='val_acc', verbose=0,\
                          save_best_only=True, save_weights_only=False,\
                          mode='auto', period=1)


history = model.fit(train_data, train_labels, epochs = 15,\
                    validation_data = (test_data, test_labels),callbacks=[monitor])

plot_acc(history)
plot_loss(history)






















