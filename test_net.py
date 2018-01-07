from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
import numpy as np


batch_size = 256
num_classes = 10   	 	
epochs = 1000
dropout_rate = 0.6
data_dir = 'E:\\MyPapers\\PYTHON\\notebooks\\TF_Practice\\train_test_data\\'
log_path = 'C:\\logs\\hyper_34_logs'
model_path = 'hyper_34_trained_model_callback.h5'



# input image dimensions
img_rows, img_cols = 5, 5
num_predictions = 40
save_dir = os.path.join(os.getcwd(), '34_saved_models')
model_name = 'keras_hyper_34_trained_model_final.h5'

X_34_train = np.load(data_dir +'X_34_train.npy')
X_34_test = np.load(data_dir +'X_34_test.npy')

y_34_train = np.load(data_dir +'y_34_train.npy')
y_34_train = y_34_train[:,2,2]

y_34_test = np.load(data_dir +'y_34_test.npy')
y_34_test = y_34_test[:,2,2]

# Convert class vectors to binary class matrices.
y_34_train = keras.utils.to_categorical(y_34_train, num_classes)
y_34_test = keras.utils.to_categorical(y_34_test, num_classes)


X_34_train /= 255
X_34_test /= 255

model = Sequential()
model.add(Conv2D(128, (1, 1), padding='same',
                 input_shape=X_34_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))

model.add(Conv2D(10, (1, 1)))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

callbacks = [
                TensorBoard(
                	log_dir=log_path, histogram_freq=10),

                EarlyStopping(
                	monitor='val_acc', 
                	mode='max',
                	patience=50,
                	verbose=1),
                
                ModelCheckpoint(model_path,
                	monitor='val_acc',
                	save_best_only=True,
                	mode='max',
                	verbose=0)
            ]

'''
EarlyStopping(
                	monitor='val_acc', 
                	mode='max',
                	patience=15,
                	verbose=1),
'''

model.fit(X_34_train, y_34_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_34_test, y_34_test),
              shuffle=True,
              callbacks=callbacks)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_34_test, y_34_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])