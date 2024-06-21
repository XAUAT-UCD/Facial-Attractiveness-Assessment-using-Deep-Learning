import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
import cv2
from tensorflow.keras.layers import BatchNormalization


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ratings = pd.read_excel(r'C:\SCUT-FBP5500_v2\SCUT-FBP5500_v2\All_Ratings.xlsx')
filenames = ratings.groupby('Filename').size().index.tolist()

labels = []

for filename in filenames:
    df = ratings[ratings['Filename'] == filename]
    score = round(df['Rating'].mean(), 2)
    labels.append({'Filename': filename, 'score': score})

labels_df = pd.DataFrame(labels)
img_width, img_height, channels = int(350*0.4), int(350*0.4), 3
sample_dir = 'C:/SCUT-FBP5500_v2/SCUT-FBP5500_v2/Images/'
nb_samples = len(os.listdir(sample_dir))
input_shape = (img_width, img_height, channels)

x_total = np.empty((nb_samples, img_width, img_height, channels), dtype=np.float32)
y_total = np.empty((nb_samples, 1), dtype=np.float32)

for i, fn in enumerate(os.listdir(sample_dir)):
    img = load_img('%s/%s' % (sample_dir, fn))
    scale_percent = 40  # percent of original size
    width = int(img_to_array(img).shape[1] * scale_percent / 100)
    height = int(img_to_array(img).shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img_to_array(img), dim, interpolation=cv2.INTER_AREA)
    x = img_to_array(img).reshape(img_height, img_width, channels)
    x = x.astype('float32') / 255.
    y = labels_df[labels_df.Filename == fn].score.values
    y = y.astype('float32')
    x_total[i] = x
    y_total[i] = y
    seed = 42

seed = 42
x_train_all, x_test, y_train_all, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=seed)

resnet = ResNet50(include_top=False, pooling='avg', input_shape=input_shape)
model = Sequential()
model.add(resnet)
model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))
model.layers[0].trainable = False


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

filepath = "{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)
callback_list = [checkpoint, reduce_learning_rate]

model.layers[0].trainable = True
model.compile(loss='mse', optimizer='adam')
history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=8,
                    epochs=30,
                    validation_data=(x_val, y_val),
                    callbacks=callback_list)

loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.title('Accuracy and Loss')
plt.plot(epochs, loss, 'blue', label='Validation loss')
plt.legend()
plt.show()
