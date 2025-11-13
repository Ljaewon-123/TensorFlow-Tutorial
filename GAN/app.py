import os
import numpy as np
from PIL import Image # pip install Pillow

paths = os.listdir('img_align_celeba/img_align_celeba')

images = []

for path in paths[0:50000]:
  numberic = PIL.Image.open('img_align_celeba/img_align_celeba/' + path).resize((64,64))
  images.append( np.array(numberic) )

images = np.array(images)
images = np.divide(images, 255)

images = images.reshape( 50000,64,64,1 )
print(images.shape)

# Discriminator

import tensorflow as tf

discriminator = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=[64,64,1]),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation='sigmoid')
]) 

generator = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4 * 4 * 256, input_shape=(100,) ), 
  tf.keras.layers.Reshape((4, 4, 256)),
  tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same'),
  tf.keras.layers.LeakyReLU(alpha=0.2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
])


GAN = tf.keras.models.Sequential([generator, discriminator])

discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
discriminator.trainable = False

GAN.compile(optimizer='adam', loss='binary_crossentropy')


# - 이미지 5만개 전처리해서 X데이터 만드는 부분
# - Generator, Discriminator 모델 만드는 부분
# - GAN 모델 만들고 Compile 하는 부분 