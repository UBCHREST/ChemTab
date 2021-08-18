# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 21:05:29 2021

@author: amol
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=45)


class DNNModel:
    def __init__(self):
        self.width = 512
        self.halfwidth = 128
        print("Parent DNNModel Instantiated")
    
    def getOptimizer(self):
        starter_learning_rate = 0.1
        end_learning_rate = 0.01
        decay_steps = 10000
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(starter_learning_rate, decay_steps, end_learning_rate, power=0.5)
        
        #opt = keras.optimizers.Adam(learning_rate=0.001)
        
        opt = keras.optimizers.Adam(learning_rate=learning_rate_fn)
                
        return opt
    
    def getIntermediateLayers(self,x):
        def add_regularized_dense_layer(x, layer_size, activation_func='relu', dropout_rate=0.25):
            x = layers.Dense(layer_size, activation=activation_func)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            return x

        def add_regularized_dense_module(x, layer_sizes, activation_func='relu', dropout_rate=0.25):
            assert len(layer_sizes)==3
            skip_input = x = add_regularized_dense_layer(x, layer_sizes[0], activation_func=activation_func, dropout_rate=dropout_rate)
            x = add_regularized_dense_layer(x, layer_sizes[1], activation_func=activation_func, dropout_rate=dropout_rate)
            x = add_regularized_dense_layer(x, layer_sizes[2], activation_func=activation_func, dropout_rate=dropout_rate)
            x = layers.Concatenate()([x, skip_input])           
            return x
  
        x = add_regularized_dense_module(x, [32,64,128])
        x = add_regularized_dense_module(x, [256,512,256])
        x = add_regularized_dense_module(x, [128,64,32])

        return x
    

    
