#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:22:49 2019

@author: guest
"""

from imageai.Prediction import ImagePrediction
import os
import pandas as pd
import numpy as np
from PIL import Image

execution_path = os.getcwd()
pred_array = np.empty((0,6),dtype=object)
prediction = ImagePrediction()
prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath(os.path.join(execution_path, "/home/guest/Documents/Test1/ImageAI-master/imageai/Prediction/Weights/SqueezeNet.h5"))
prediction.loadModel()
TEST_PATH = '/home/guest/Documents/Aikomi'


for img in os.listdir(TEST_PATH):
    if img.endswith('.jpg'):
        image = Image.open(os.path.join(TEST_PATH, img))
        image = image.convert("RGB")
        image = np.array(image, dtype=np.uint8)
        # It should probably be img instead of "5.jpg"
        predictions, probabilities = prediction.predictImage(os.path.join(TEST_PATH, img), result_count=5)
        temprow = np.zeros((1,pred_array.shape[1]),dtype=object)
        temprow[0,0] = img
        for i in range(len(predictions)):
            temprow[0,i+1] = (predictions[i],probabilities[i])
        #temprow=np.array(temprow).reshape(-1,6)
        #for eachPrediction, eachProbability in zip(predictions, probabilities):
            #print(eachPrediction , " : " , eachProbability)
        pred_array = np.append(pred_array,temprow,axis=0)
df = pd.DataFrame(data=pred_array,columns=['File_name','Tag_1','Tag_2','Tag_3','Tag_4','Tag_5'])  
print(df)
df.to_csv('Image_tags.csv')



