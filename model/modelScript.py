import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import numpy as np
import pickle as pkl
from models import build_base_convnet_model, load_image_data, fit_model, build_inception_model, build_resnet_model

# Ignore the following if not using Windows' gpu
import os
# Add directory for NVIDIA gpu. Ignore if not Windows
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# Confirm tensorflow running on gpu. Can ignore
print(tf.config.list_physical_devices('GPU'))


## Base CovNet Model
base_mod = build_base_convnet_model()
train_data, validation_data, test_data = load_image_data('../data/COVID-19_Radiography_Dataset/TwoClasses/split')
covnet_model = fit_model(base_mod, train_data, validation_data)
results = covnet_model.evaluate(test_data)

## Base InceptionV3 Model
transfer_learning_mod = build_inception_model()
inception_model = fit_model(transfer_learning_mod, train_data, validation_data)
results = inception_model.evaluate(test_data)

## Base ResNet Model
transfer_learning_mod = build_resnet_model()
resnet_model = fit_model(transfer_learning_mod, train_data, validation_data)
results = resnet_model.evaluate(test_data)









## Final Model Pickle

# Create full train data
fulltrain = train_data.concatenate(validation_data)

# Fitting final model
model = resnet_model
final_model = model.fit(fulltrain)

# Pickling final model
filename = 'final_model.pkl'
pkl.dump(final_model, open(filename, 'wb'))