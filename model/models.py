from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from keras.applications import InceptionV3, ResNet50

# function to build the conv net base


# complete this function
def build_base_convnet_model():
    """Re-create the model from the first prompt, but with a different input shape"""
    
    # Return this variable
    model = None
    
    # YOUR CODE HERE
    inputs = keras.Input(shape = (299, 299, 3))
    x = keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu')(inputs)
    x = keras.layers.MaxPooling2D(pool_size = 2)(x)
    x = keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu')(x)
    x = keras.layers.MaxPooling2D(pool_size = 2)(x)
    x = keras.layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu')(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def load_image_data(base_path: str) -> tuple:
    """Write a function that accepts a base path that contains all of the directories, and creates training,
    validation and test sets"""
    
    # Return these variables from the function
    train_data = keras.utils.image_dataset_from_directory(f'{base_path}/train', 
                                                          image_size = (299, 299),
                                                          batch_size = 32)

    validation_data = keras.utils.image_dataset_from_directory(f'{base_path}/val', 
                                                          image_size = (299, 299),
                                                          batch_size = 32)
    
    test_data = keras.utils.image_dataset_from_directory(f'{base_path}/test', 
                                                          image_size = (299, 299),
                                                          batch_size = 32)
    
    # YOUR CODE HERE
    
    
    return train_data, validation_data, test_data

def fit_model(model, train_set, validation_set):
    """Fit a model with the above stated criteria"""
    early_stopping = keras.callbacks.EarlyStopping(patience = 10)
    
    # YOUR CODE HERE
    model.fit(train_set, 
              validation_data = validation_set, 
              callbacks = [early_stopping], 
              epochs = 500)
    
    return model

def build_inception_model():
    
    # return this variable
    model = None

    model_input = keras.Input(shape = (299, 299, 3))
    
    base_model = InceptionV3(input_shape = (299, 299, 3), weights='imagenet', include_top=False)

    # make the weights in the base model non-trainable
    for layer in base_model.layers:
      layer.trainable = False

    # combine the base model with a dense layer and output layer for the 10 classes
    # the preprocess_input transforms input data according to how the model was trained
    
    x = keras.applications.inception_v3.preprocess_input(model_input)
    x = base_model(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation = 'relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model(model_input, output)

    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

def build_resnet_model():
    
    # return this variable
    model = None

    model_input = keras.Input(shape = (299, 299, 3))
    
    base_model = ResNet50(input_shape = (299, 299, 3), weights='imagenet', include_top=False)

    # make the weights in the base model non-trainable
    for layer in base_model.layers:
      layer.trainable = False

    # combine the base model with a dense layer and output layer for the 10 classes
    # the preprocess_input transforms input data according to how the model was trained
    
    x = keras.applications.resnet50.preprocess_input(model_input)
    x = base_model(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation = 'relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model(model_input, output)

    model.compile(loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model