# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:01:41 2020

@author: Santosh Sah
"""

import pickle
from keras.preprocessing.image import ImageDataGenerator

"""
Import training set.
"""
def importConvolutionalNeuralNetworkTrainingSet(convolutionalNeuralNetworkTrainingSetPath):
    
    training_datagenerator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    
    training_set = training_datagenerator.flow_from_directory(convolutionalNeuralNetworkTrainingSetPath, target_size=(64, 64), batch_size=32, class_mode="binary")
    
    return training_set

"""
Import testing set.
"""
def importConvolutionalNeuralNetworkTestingSet(convolutionalNeuralNetworkTestingSetPath):
    
    testing_datagenerator = ImageDataGenerator(rescale=1./255)
    
    testing_set = testing_datagenerator.flow_from_directory(convolutionalNeuralNetworkTestingSetPath, target_size=(64,64), batch_size=32, class_mode="binary")
    
    return testing_set

    
"""
Save training dataset
"""
def saveConvolutionalNeuralNetworkTrainingDataset(training_set):
    
    #Write training_set in a picke file
    with open("training_set.pkl",'wb') as training_set_Pickle:
        pickle.dump(training_set, training_set_Pickle, protocol = 2)
    
"""
Save testing dataset
"""
def saveConvolutionalNeuralNetworkTestingDataset(testing_set):
        
    #Write testing_set in a picke file
    with open("testing_set.pkl",'wb') as testing_set_Pickle:
        pickle.dump(testing_set, testing_set_Pickle, protocol = 2)
    
"""
Save ConvolutionalNeuralNetworkModel as a pickle file.
"""
def saveConvolutionalNeuralNetworkModel(convolutionalNeuralNetworkModel):
    
    #Write ConvolutionalNeuralNetworkModel as a picke file
    with open("ConvolutionalNeuralNetworkModel.pkl",'wb') as ConvolutionalNeuralNetworkModel_Pickle:
        pickle.dump(convolutionalNeuralNetworkModel, ConvolutionalNeuralNetworkModel_Pickle, protocol = 2)

"""
read ConvolutionalNeuralNetworkModel from pickle file
"""
def readConvolutionalNeuralNetworkModel():
    
    #load ConvolutionalNeuralNetworkModel model
    with open("ConvolutionalNeuralNetworkModel.pkl","rb") as ConvolutionalNeuralNetworkModel:
        convolutionalNeuralNetworkModel = pickle.load(ConvolutionalNeuralNetworkModel)
    
    return convolutionalNeuralNetworkModel

"""
read ConvolutionalNeuralNetworkTraining from pickle file
"""
def readConvolutionalNeuralNetworkTraining():
    
    #load ConvolutionalNeuralNetworkTraining
    with open("training_set.pkl","rb") as training_set_pickle:
        training_set = pickle.load(training_set_pickle)
    
    return training_set

"""
read testing_set from pickle file
"""
def readConvolutionalNeuralNetworkTesting():
    
    #load testing_set
    with open("testing_set.pkl","rb") as testing_set_pickle:
        testing_set = pickle.load(testing_set_pickle)
    
    return testing_set

"""
Save ConvolutionalNeuralNetworkModel with extra convolution layer as a pickle file.
"""
def saveConvolutionalNeuralNetworkWithExtraConvolutionLayerModel(convolutionalNeuralNetworkWithExtraConvolutionLayerModel):
    
    #Write ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel as a picke file
    with open("ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel.pkl",'wb') as ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel_Pickle:
        pickle.dump(convolutionalNeuralNetworkWithExtraConvolutionLayerModel, ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel_Pickle, protocol = 2)

"""
read ConvolutionalNeuralNetworkModel extra convolution layer from pickle file
"""
def readConvolutionalNeuralNetworkWithExtraConvolutionLayerModel():
    
    #load ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel model
    with open("ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel.pkl","rb") as ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel:
        convolutionalNeuralNetworkWithExtraConvolutionLayerModel = pickle.load(ConvolutionalNeuralNetworkWithExtraConvolutionLayerModel)
    
    return convolutionalNeuralNetworkWithExtraConvolutionLayerModel
