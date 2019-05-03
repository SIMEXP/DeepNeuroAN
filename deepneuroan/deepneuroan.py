#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:55:12 2018

@author: ltetrel
"""
import os
import tensorflow as tf

def data_read(path):
    

def main():
    
    data_read(path)
    
    leNet5 = tf.keras.models.Sequential()

    # Feature extraction
    leNet5.add(tf.keras.layers.Conv2D(input_shape=in_shape, filters=20, kernel_size=(5, 5), activation=tf.nn.relu))
    leNet5.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    leNet5.add(tf.keras.layers.Conv2D(input_shape=in_shape, filters=50, kernel_size=(3, 3), activation=tf.nn.relu))
    leNet5.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Classification
    #leNet5.add(tf.keras.layers.Dropout(0.25))
    leNet5.add(tf.keras.layers.Flatten())
    leNet5.add(tf.keras.layers.Dense(units=500, activation=tf.nn.relu))
    #leNet5.add(tf.keras.layers.Dropout(0.5))
    leNet5.add(tf.keras.layers.Dense(units=nb_classes, activation=tf.nn.softmax))
    
    # Optimizer
    opt = tf.keras.optimizers.SGD(lr=0.01)
    leNet5.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Training
    leNet5.fit(x_train, y_train, batch_size=128, epochs=50, verbose=1)
    
    # Testing
    leNet5.evaluate(x_test, y_test, batch_size=128, verbose=1)
    
    # Saving the model
    sess = tf.keras.backend.get_session()
    tf.train.Saver().save(sess, "/notebooks/yu_gpu_cpu_profile/LeNetVisu/LeNet5")
    
    return 0

if __name__ == '__main__':
    main()
