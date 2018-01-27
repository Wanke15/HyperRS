from __future__ import print_function
import keras
from keras.models import Sequential
from keras.models import load_model
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
log_path = 'C:\\logs\\retrain_5_2_6'
save_dir = "C:\\Users\\Administrator\\Documents\\Mynotebooks\\models\\retrain_5_2_6"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_callback = os.path.join(save_dir, "hyper_trained_model_callback.h5")
model_final = os.path.join(save_dir, "hyper_trained_model_final.h5")


# input image dimensions
img_rows, img_cols = 5, 5
num_predictions = 20

X = np.load(data_dir +'X_train.npy')

y = np.load(data_dir +'y_train.npy')
#print(y.shape)

tr_X, te_X, tr_y, te_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert class vectors to binary class matrices.
tr_y = keras.utils.to_categorical(tr_y, num_classes)
te_y = keras.utils.to_categorical(te_y, num_classes)
print(tr_y.shape)

tr_X /= 255
te_X /= 255

model = load_model("../models/v5_2_6/hyper_trained_model_callback.h5")

# initiate RMSprop optimizer
# opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
opt = keras.optimizers.Adam(lr=0.000000001, decay=1e-10)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

callbacks = [
                TensorBoard(
                  log_dir=log_path, histogram_freq=100),

                EarlyStopping(
                  monitor='loss', 
                  mode='auto',
                  patience=30,
                  verbose=1),

                #TerminateOnNaN(),

                ReduceLROnPlateau(
                    monitor='loss', 
                    factor=0.33,
                    patience=3,
                    min_lr=1e-12),
                
                ModelCheckpoint(model_callback,
                  monitor='val_acc',
                  save_best_only=True,
                  mode='auto',
                  verbose=0)
            ]


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
