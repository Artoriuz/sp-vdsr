import numpy as np
import tensorflow as tf
import cv2
import glob
import scipy.io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def gradient_sensitive_loss(y_true, y_pred):
    dY = tf.image.sobel_edges(y_true)
    dX = tf.image.sobel_edges(y_pred)
    G = tf.sqrt(tf.square(dY[:,:,:,:,0]) + tf.square(dY[:,:,:,:,1]))
    M = G / tf.math.reduce_max(G)
    return tf.keras.losses.MSE(M * y_true, M * y_pred) + 2 * tf.keras.losses.MSE((1.0 - M) * y_true, (1.0 - M) * y_pred)

def ssim_loss(y_true, y_pred):
    ssim_1 = tf.image.ssim(y_true, y_pred, max_val=255)
    return 1 - tf.reduce_mean(ssim_1)

def ssim_mse_loss(y_true, y_pred):
    ssim_1 = tf.image.ssim(y_true, y_pred, max_val=255)
    mse_1 = tf.keras.losses.MSE(y_true, y_pred)
    return (1 - tf.reduce_mean(ssim_1)) * mse_1

def msssim_mse_loss(y_true, y_pred):
    msssim_1 = tf.image.ssim_multiscale(y_true, y_pred, max_val=255)
    mse_1 = tf.keras.losses.MSE(y_true, y_pred)
    return (1 - tf.reduce_mean(msssim_1)) * mse_1

def psnr_metric(y_true, y_pred):
    psnr_1 = tf.image.psnr(y_true, y_pred, max_val=255)
    return tf.reduce_mean(psnr_1)

def ssim_metric(y_true, y_pred):
    ssim_1 = tf.image.ssim(y_true, y_pred, max_val=255)
    return tf.reduce_mean(ssim_1)

def msssim_metric(y_true, y_pred):
    msssim_1 = tf.image.ssim_multiscale(y_true, y_pred, max_val=255)
    return tf.reduce_mean(msssim_1)

filelist1 = glob.glob('./input/*.png')
low_res = []
for myFile in filelist1:
    image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
    low_res.append (image)
low_res = np.array(low_res).astype(np.float32).reshape(32,540,960,1)

filelist2 = glob.glob('./interpolated/*.png')
interpolated = []
for myFile in filelist2:
    image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
    interpolated.append (image)
interpolated = np.array(interpolated).astype(np.float32).reshape(32,1080,1920,1)

filelist3 = glob.glob('./output/*.png')
high_res = []
for myFile in filelist3:
    image = cv2.imread(myFile, cv2.IMREAD_GRAYSCALE)
    high_res.append (image)
high_res = np.array(high_res).astype(np.float32).reshape(32,1080,1920,1)

# Build the model.
low_res_input = tf.keras.Input(shape=(540,960,1))
interpolated_input = tf.keras.Input(shape=(1080,1920,1))
x0 = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)(interpolated_input)
x1 = tf.keras.layers.experimental.preprocessing.Rescaling(1.0/255)(low_res_input)
x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(x1)
x1 = tf.keras.layers.LeakyReLU(alpha=0.3)(x1)
x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(x1)
x1 = tf.keras.layers.LeakyReLU(alpha=0.3)(x1)
x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(x1)
x1 = tf.keras.layers.LeakyReLU(alpha=0.3)(x1)
x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(x1)
x1 = tf.keras.layers.LeakyReLU(alpha=0.3)(x1)
x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same')(x1)
x1 = tf.keras.layers.LeakyReLU(alpha=0.3)(x1)
x1 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, padding='same')(x1)
x1 = tf.nn.depth_to_space(x1, 2)
x1 = tf.keras.layers.Add()([x1, x0])
x1 = tf.keras.layers.ReLU(max_value=1)(x1)
outputs = tf.keras.layers.experimental.preprocessing.Rescaling(255)(x1)
model = tf.keras.Model(inputs=[low_res_input, interpolated_input], outputs=outputs)

opt = tf.keras.optimizers.Adam(learning_rate=0.005)

def scheduler(epoch, lr):
  if epoch < 100:
    return 0.01
  elif epoch < 3000:
    return 0.005
  else:
    return 0.00025

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Compile the model.
model.compile(optimizer=opt, loss='mse', metrics=psnr_metric)
model.summary()

# Train the model.
history = model.fit([low_res, interpolated], high_res, epochs=3600, callbacks=[callback], batch_size=32, verbose=1)

# Make a prediction
predictions = model.predict([low_res, interpolated])
pred_data = ((predictions).astype(np.uint8)).reshape(160,1080,1920)
pred_data = ((predictions).astype(np.uint8)).reshape(160,540,960)
scipy.io.savemat('./predictions.mat', {'pred_data':pred_data})

