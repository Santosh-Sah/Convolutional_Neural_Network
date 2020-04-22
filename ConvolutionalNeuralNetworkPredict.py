# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:03:31 2020

@author: Santosh Sah
"""

import numpy as np
from keras.preprocessing import image
from ConvolutionalNeuralNetworkUtils import readConvolutionalNeuralNetworkWithExtraConvolutionLayerModel,readConvolutionalNeuralNetworkModel 

def predict():
    
    convolutionalNeuralNetworkWithExtraConvolutionLayerModel = readConvolutionalNeuralNetworkWithExtraConvolutionLayerModel()
    ConvolutionalNeuralNetworkModel = readConvolutionalNeuralNetworkModel()
    
    #test_image = image.load_img("single_prediction/cat_or_dog_1.jpg", target_size=(64,64))
    test_image = image.load_img("single_prediction/cat_or_dog_2.jpg", target_size=(64,64))
    #test_image = image.load_img("single_prediction/cat.11.jpg", target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    result1 = convolutionalNeuralNetworkWithExtraConvolutionLayerModel.predict(test_image)
    result2 = ConvolutionalNeuralNetworkModel.predict(test_image)
        
    if result1[0][0] == 1:
        print("image is of Dog")
    else:
        print("image is of Cat")
        
    if result2[0][0] == 1:
        print("image is of Dog")
    else:
        print("image is of Cat")

if __name__ == "__main__":
    predict()