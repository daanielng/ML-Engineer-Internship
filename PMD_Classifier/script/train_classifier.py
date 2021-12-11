import glob
import os
import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet101, ResNet152, ResNet50, Xception, ResNet50V2, DenseNet201, DenseNet169, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import SGD, Adam


####################
# Helper Functions #
####################
#locates latest h5 file in current weights folder
def find_latest_h5(path):
    latest_epoch = 0
    h5_file = ''
    for h5_path in glob.glob(path + '\*'):
        filename = h5_path.split('\\')[-1]
        epoch = int(filename.split('weights.')[1].split('-')[0])
        
        if latest_epoch < epoch:
            latest_epoch = epoch
            h5_file = filename
    return [h5_file, latest_epoch]
    
#locates h5 file in the h5_to_uff folder: for transfer learning
def find_h5_file(folder):
    for filepath in glob.glob(folder + '\*'):
        filename = filepath.split('\\')[-1]
        if '.h5' in filename:
            return filename        



########################
# INPUTS FROM TERMINAL #
########################
today_date = input("Enter Today's Date: ") #to obtain correct data directory
model_name = input("Enter Model's Name (e.g PMD_V31): ") 
num_classes = input("Enter number of classes (e.g 5): ") 
gpu_num = input("Enter GPU Device number (e.g 0): ")
resume_training = input("Resume Training? (True/False): ") # continue training for current model
if resume_training == 'False':
    transfer_learning = input("Apply Transfer Learning for new training? (True/False): ") # if true, use previous trained model to train new model
    if transfer_learning == 'True':
        prev_model_version = input("Which latest model to load from? (e.g 49): ")
        prev_model_version = int(prev_model_version)



###############
# Directories #
###############
train_dir = r'D:\Daniel\PMD\model_training\\final_data\combined_data_{}\\train'.format(today_date)
valid_dir = r'D:\Daniel\PMD\model_training\\final_data\combined_data_{}\\valid'.format(today_date)
logs_dir = r'D:\Daniel\PMD\model_training\logs_dan\{}'.format(model_name)
weights_dir = r'D:\Daniel\PMD\model_training\weights_dan\{}'.format(model_name)
if resume_training == 'False':
    if transfer_learning == 'True':
        previous_models_dir = r'D:\Daniel\PMD\scripts\H5 to UFF\pmd_v{}'.format(prev_model_version)

# Create weights and logs folder if not created
if not os.path.exists(weights_dir):
    print("Weights dir not found... creating a new one")
    os.mkdir(weights_dir)
if not os.path.exists(logs_dir):
    print("Logs dir not found... creating a new one")
    os.mkdir(logs_dir)


########################
# Initialize variables #
########################
num_classes = int(num_classes)
set_batch_size = 32
width = 110
height = 260

# Select GPU Device
if gpu_num != None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num


# Load Model
if resume_training == 'True': #resume training of current model
    latest_weights, latest_epoch = find_latest_h5(weights_dir)
    weights_path = weights_dir + '\\' + latest_weights
    
    start_epoch = latest_epoch
    model = load_model(weights_path)
    print('~~~~~~~~ RESUMING TRAINING ~~~~~~~')
    status = "Model loaded with weights {}".format(weights_path.split('/')[-1])

elif transfer_learning == 'True': # new training with previously trained model
    weights = find_h5_file(previous_models_dir)
    weights_path = previous_models_dir + '\\' + weights
    
    start_epoch = 0
    model = load_model(weights_path)
    print('~~~~~~~~ TRANSFER LEARNING ~~~~~~~')
    status = "Model loaded with weights {}".format(weights_path.split('/')[-1])

else: # new training with selected model architecture
    input_tensor = Input(shape=(height, width, 3))
    base_model = DenseNet201(input_tensor=input_tensor, include_top=False, weights='imagenet')

    for layer in base_model.layers[:]: # let each layer in model to be trainable
        layer.trainable = True
        
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    start_epoch = 0
    model = Model(inputs=base_model.input, outputs=x)


print('Model Loaded')


# Loss Functions
sgd = SGD(learning_rate=1e-5, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(learning_rate=1e-5)



#################
# Compile Model #
#################
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


####################
# Data Preparation #
####################
train_datagen = ImageDataGenerator(rescale = 1./255,
                                #    rotation_range = 40,
                                   zoom_range = 0.2,
                                   width_shift_range=0.2,
                                #    height_shift_range=0.2,
                                   horizontal_flip = True,
                                   brightness_range = [0.8, 1.0],
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = (height, width),
                                                 batch_size = set_batch_size,
                                                 class_mode = 'categorical')

valid_set = test_datagen.flow_from_directory(valid_dir,
                                            target_size = (height, width),
                                            batch_size = set_batch_size,
                                            class_mode = 'categorical')


#############
# Callbacks #
#############

# Model Checkpointer
checkpointer = ModelCheckpoint(filepath = os.path.join(weights_dir,"weights.{epoch:02d}-{val_accuracy:.4f}-{val_loss:.4f}.h5"), verbose=1, save_best_only=False)


# Logging with Tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(
                                            log_dir=logs_dir,
                                            histogram_freq=0,
                                            write_graph=True,
                                            write_images=True,
                                            update_freq="epoch",
                                            profile_batch=2,
                                            embeddings_freq=0,
                                            embeddings_metadata=None,  
											)


# LR Scheduler: Reduce Learning Rate when val_loss does not decrease
def scheduler(epoch, lr):
    if epoch<5:
        return lr
    else:
        return lr*tf.math.exp(-0.1)

reduce_lr_by_schedule = LearningRateScheduler(scheduler, verbose=1)


# LR on Plateau 
reduce_lr_on_plateau = ReduceLROnPlateau(
                                        monitor='val_loss', 
                                        factor=0.1, 
                                        patience=5, 
                                        verbose=1,
                                        mode='min', 
                                        min_delta=0.0001, 
                                        cooldown=0, 
                                        min_lr=0,
)


# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.001, patience = 30, mode='min')




##################
# Model Training #
##################
hist = model.fit(training_set,
                steps_per_epoch = (24000//set_batch_size),
                epochs = 200, # how many epochs to train model
                validation_data = valid_set,
                validation_steps = (12234//set_batch_size),
                workers = 4,
                initial_epoch = start_epoch,
                callbacks = [checkpointer,tensorboard, reduce_lr_on_plateau]
                )
