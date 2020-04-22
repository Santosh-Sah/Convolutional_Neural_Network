# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:03:56 2020

@author: Santosh Sah
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from ConvolutionalNeuralNetworkUtils import (importConvolutionalNeuralNetworkTrainingSet, importConvolutionalNeuralNetworkTestingSet,
                                             saveConvolutionalNeuralNetworkModel, saveConvolutionalNeuralNetworkWithExtraConvolutionLayerModel)

"""
Train ConvolutionalNeuralNetwork model 
"""
def trainConvolutionalNeuralNetworkModel():
    
    #initializing CNN
    convolutionalNeuralNetwork = Sequential()
    
    #Convolution
    convolutionalNeuralNetwork.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation="relu"))
    
    #Pooling
    convolutionalNeuralNetwork.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Flattening
    convolutionalNeuralNetwork.add(Flatten())
    
    #Full connection
    convolutionalNeuralNetwork.add(Dense(output_dim = 128, activation="relu"))
    convolutionalNeuralNetwork.add(Dense(output_dim = 1, activation="sigmoid"))
    
    #Compile CNN
    convolutionalNeuralNetwork.compile(optimizer="adam", loss="binary_crossentropy", metrics= ["accuracy"])
    
    #Image agumentation, also called preprocessing of the image
    training_set = importConvolutionalNeuralNetworkTrainingSet("dataset/training_set")
    testing_set = importConvolutionalNeuralNetworkTestingSet("dataset/test_set") 
    
    #fit the model with the dataset
    convolutionalNeuralNetwork.fit_generator(training_set, samples_per_epoch=8000, nb_epoch = 25, validation_data=testing_set, nb_val_samples=2000)   
    
    #saving the model in pickle files
    saveConvolutionalNeuralNetworkModel(convolutionalNeuralNetwork)
    
    """
    accuracy on the test set is around 75%
    """

"""
Train ConvolutionalNeuralNetwork with extra convolution and pooled layer model 
"""
def trainConvolutionalNeuralNetworkWithExtraConvolutionLayerModel():
    
    #initializing CNN
    convolutionalNeuralNetwork = Sequential()
    
    #Convolution
    convolutionalNeuralNetwork.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation="relu"))
    
    #Pooling
    convolutionalNeuralNetwork.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Extra Convolution
    convolutionalNeuralNetwork.add(Convolution2D(32, 3, 3, activation="relu"))
    
    #Extra Pooling
    convolutionalNeuralNetwork.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Flattening
    convolutionalNeuralNetwork.add(Flatten())
    
    #Full connection
    convolutionalNeuralNetwork.add(Dense(output_dim = 128, activation="relu"))
    convolutionalNeuralNetwork.add(Dense(output_dim = 1, activation="sigmoid"))
    
    #Compile CNN
    convolutionalNeuralNetwork.compile(optimizer="adam", loss="binary_crossentropy", metrics= ["accuracy"])
    
    #Image agumentation, also called preprocessing of the image
    training_set = importConvolutionalNeuralNetworkTrainingSet("dataset/training_set")
    testing_set = importConvolutionalNeuralNetworkTestingSet("dataset/test_set") 
    
    #fit the model with the dataset
    convolutionalNeuralNetwork.fit_generator(training_set, samples_per_epoch=8000, nb_epoch = 25, validation_data=testing_set, nb_val_samples=2000)   
    
    #saving the model in pickle files
    saveConvolutionalNeuralNetworkWithExtraConvolutionLayerModel(convolutionalNeuralNetwork)
    
    """
    accuracy on the test set is around 75%
    """
    
if __name__ == "__main__":
    #trainConvolutionalNeuralNetworkModel()
    trainConvolutionalNeuralNetworkWithExtraConvolutionLayerModel()