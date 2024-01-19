import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import load_model
from PIL import Image


X_train = np.load('datasets/X_train.npy')
X_test = np.load('datasets/X_test.npy')
y_train = np.load('datasets/y_train.npy')
y_test = np.load('datasets/y_test.npy')

model = load_model("checkpoints/model_new-38.2-acc.h5")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"])

fig, axs = plt.subplots(3, 3, figsize=(12, 12))

axs[0][0].imshow(np.stack(X_test[50]))
axs[0][0].set_title('Raw (Bhuvan) 1')
axs[0][1].imshow(np.stack(X_test[51]))
axs[0][1].set_title('Raw (Bhuvan) 2')
axs[0][2].imshow(np.stack(X_test[52]))
axs[0][2].set_title('Raw (Bhuvan) 3')

axs[1][0].imshow(y_test[50])
axs[1][0].set_title('Normalized (Bhoonidhi) 1')
axs[1][1].imshow(y_test[51])
axs[1][1].set_title('Normalized (Bhoonidhi) 2')
axs[1][2].imshow(y_test[52])
axs[1][2].set_title('Normalized (Bhoonidhi) 3')

# Predict the images
predicted = model.predict(X_train[50:53])

axs[2][0].imshow(predicted[0])
axs[2][0].set_title('Predicted 1')
axs[2][1].imshow(predicted[1])
axs[2][1].set_title('Predicted 2')
axs[2][2].imshow(predicted[2])
axs[2][2].set_title('Predicted 3')

for ax in axs.flatten():
    ax.axis('off')

plt.show()
