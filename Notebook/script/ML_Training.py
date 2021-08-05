#encoding: utf-8
#有上面這行才能用中文註解
"""
-----------------------------------------------------------
""python3""
\033[3;32m ML_Training.py \033[0;m
\033[3;31m Usage: ML_Training(input_train,input_test,source_train,source_test) \033[0;m
\033[3;31m        Trained Model will be stroed in "Model" directory \033[0;m
\033[3;31m Usage: Load_ML_ML_Training(input_train,input_test,source_train,source_test,model_path,EPOCH=250,BATCH=256)\033[0;m
\033[3;31m        Load Model for snd training \033[0;m
\033[3;31m        Trained Model will be stroed in "Model" directory \033[0;m


-----------------------------------------------------------
"""
print(__doc__)
# from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import time
# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout #, BatchNormalization, Activation
from tensorflow.keras import metrics, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, CSVLogger
from tensorflow.keras.optimizers import Adam , SGD , Adagrad
import myfunction as my_fn
import os
import sys




def ML_Training(input_train,input_test,source_train,source_test,EPOCH=100):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    ticks_1 = time.time()
    ######################################################################################################
    """
    Create a directory to store model.
    """
    if os.path.exists("./Model_denoise") == 0:
        os.mkdir("./Model_denoise")
    
    
    model_generator = Sequential(name = 'Sequential')
    activ = "elu"
    model_generator.add(Conv1D(activation=activ, input_shape=(input_train.shape[1], 84), filters=512, kernel_size=1, name = "Conv1D_input"))
    model_generator.add(Conv1D(activation=activ, filters=512, kernel_size=1, name = "Conv1D_1"))
    model_generator.add(Conv1D(activation=activ, filters=256, kernel_size=1, name = "Conv1D_2"))
    model_generator.add(Conv1D(activation=activ, filters=256, kernel_size=1, name = "Conv1D_3"))
    model_generator.add(Conv1D(activation=activ, filters=128, kernel_size=1, name = "Conv1D_4"))
    model_generator.add(MaxPooling1D(pool_size=2))
    model_generator.add(Flatten())

    model_generator.add(Dense(128, activation=activ, kernel_initializer='glorot_uniform', name = "Dense_1"))
    model_generator.add(Dense(128, activation=activ, kernel_initializer='glorot_uniform', name = "Dense_2"))
    model_generator.add(Dense(64, activation=activ, kernel_initializer='glorot_uniform', name = "Dense_3"))
    model_generator.add(Dense(64, activation=activ, kernel_initializer='glorot_uniform', name = "Dense_4"))
    model_generator.add(Dense(32, activation=activ, kernel_initializer='glorot_uniform', name = "Dense_5"))
    model_generator.add(Dense(source_train.shape[1], activation="linear", kernel_initializer='glorot_uniform', name = "Dense_out"))
    adam = Adam(lr=0.00008, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model_generator.compile(loss='logcosh',optimizer=adam, metrics=['accuracy','mse','mae','mape'])
    model_generator.summary()

    check_list=[]
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./Model_denoise/log_test")
#     lrate = LearningRateScheduler(my_fn.step_decay,verbose=1)
    checkpoint = ModelCheckpoint(
            filepath='./Model_denoise/CR_ML_Checkpoint.h5',
            save_best_only=True,
            verbose=1)
    csv_logger = CSVLogger('./Model_denoise/training_log.csv')
#     earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
#                 verbose=1, mode='min', baseline=None, restore_best_weights=True)
    check_list.append(tensorboard_callback)
    check_list.append(checkpoint)
    check_list.append(csv_logger)
    # check_list.append(lrate)
    # check_list.append(earlystop)
    
    epoch = EPOCH
#     epoch = 100
    batch_s = 32
    training_history = model_generator.fit(
        input_train,
        source_train,
        epochs = epoch,
        batch_size = batch_s, 
        validation_data=(input_test, source_test),
        callbacks=check_list,
        verbose=1
        )

    loss = model_generator.evaluate(input_test, source_test, verbose=0)
    # print(training_history.history.keys())
    model_generator.save("./Model_denoise/CR_ML.h5")

    loss = model_generator.evaluate(input_test, source_test,verbose=0)
    print("{}: {:.5f}".format(model_generator.metrics_names[1],loss[1]))
    print("{}: {:.5f}".format(model_generator.metrics_names[2],loss[2]))
    print("{}: {:.5f}".format(model_generator.metrics_names[3],loss[3]))
    print("{}: {:.5f}".format(model_generator.metrics_names[4],loss[4]))
    #######################################################################################################    
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    print("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))
    
    
def Load_ML_ML_Training(input_train,input_test,source_train,source_test,model_path, EPOCH = 250, BATCH = 256):
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    ticks_1 = time.time()
    ######################################################################################################
    """
    Create a directory to store model.
    """
    if os.path.exists("./Model") == 0:
        os.mkdir("./Model")
    try:
        model_generator = load_model(model_path)
    except :
        print("Please Check the Model Path!")
        sys.exit(1)
    
    model_generator.summary()

    check_list=[]
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./Model/log_test_2nd")
#     lrate = LearningRateScheduler(my_fn.step_decay,verbose=1)
    checkpoint = ModelCheckpoint(
            filepath='./Model/CR_ML_Checkpoint_2nd.h5',
            save_best_only=True,
            verbose=1)
    csv_logger = CSVLogger('./Model/training_log_2nd.csv')
#     earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20,
#                 verbose=1, mode='min', baseline=None, restore_best_weights=True)
    check_list.append(tensorboard_callback)
    check_list.append(checkpoint)
    check_list.append(csv_logger)
    # check_list.append(lrate)
    # check_list.append(earlystop)

    epoch = EPOCH
    batch_s = BATCH 
    training_history = model_generator.fit(
        input_train,
        source_train,
        epochs = epoch,
        batch_size = batch_s, #16
        validation_data=(input_test, source_test),
    #     validation_split=0.2,
        callbacks=check_list,
        verbose=1
        )

    loss = model_generator.evaluate(input_test, source_test, verbose=0)
    # print(training_history.history.keys())
    model_generator.save("./Model/CR_ML_2nd.h5")

    loss = model_generator.evaluate(input_test, source_test,verbose=0)
    print("{}: {:.5f}".format(model_generator.metrics_names[1],loss[1]))
    print("{}: {:.5f}".format(model_generator.metrics_names[2],loss[2]))
    print("{}: {:.5f}".format(model_generator.metrics_names[3],loss[3]))
    print("{}: {:.5f}".format(model_generator.metrics_names[4],loss[4]))
    #######################################################################################################    
    ticks_2 = time.time()
    totaltime =  ticks_2 - ticks_1
    print("\033[3;33mTime consumption : {:.4f} min\033[0;m".format(totaltime/60.))