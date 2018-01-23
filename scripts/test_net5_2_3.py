from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
import numpy as np
np.random.seed(0)

from sklearn.model_selection import train_test_split


batch_size = 256
num_classes = 10      
epochs = 100000
dropout_rate = 0.6
data_dir = "C:\\Users\\Administrator\\Documents\\Mynotebooks\\data\\ready\\"
log_path = 'C:\\logs\\v5_2_3'
save_dir = "C:\\Users\\Administrator\\Documents\\Mynotebooks\\models\\v5_2_3"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_callback = os.path.join(save_dir, "hyper_trained_model_callback.h5")
model_final = os.path.join(save_dir, "hyper_trained_model_final.h5")


# input image dimensions
img_rows, img_cols = 5, 5
num_predictions = 20

X = np.load(data_dir +'X_train.npy')

y = np.load(data_dir +'y_train.npy')
print(y.shape)

tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert class vectors to binary class matrices.
tr_y = keras.utils.to_categorical(tr_y, num_classes)
te_y = keras.utils.to_categorical(te_y, num_classes)


tr_X /= 255
te_X /= 255

model = Sequential()
model.add(Conv2D(128, (1, 1), padding='valid',
                 input_shape=tr_X.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Dropout(dropout_rate))
'''
model.add(Conv2D(32, (1, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout_rate))
'''
model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Dropout(dropout_rate))

model.add(Conv2D(10, (1, 1)))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())

# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
opt = keras.optimizers.Adam(lr=0.000001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

callbacks = [
                TensorBoard(
                  log_dir=log_path, histogram_freq=100),

                EarlyStopping(
                  monitor='val_acc', 
                  mode='max',
                  patience=100,
                  verbose=1),

                #TerminateOnNaN(),

                ReduceLROnPlateau(
                    monitor='loss', 
                    factor=0.7,
                    patience=5,
                    min_lr=0.00000001),
                
                ModelCheckpoint(model_callback,
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

model.fit(tr_X, tr_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(te_X, te_y),
              shuffle=True,
              callbacks=callbacks)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
#model_final = os.path.join(save_dir, model_final)
model.save(model_final)
print('Saved trained model at %s ' % model_final)

# Score trained model.
scores = model.evaluate(te_X, te_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
