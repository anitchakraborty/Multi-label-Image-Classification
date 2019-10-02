from imageai.Prediction import ImagePrediction
import os
import pandas as pd
import numpy as np
from PIL import Image
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

execution_path = os.getcwd()
TEST_PATH = '/home/guest/Documents/Aikomi'

prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "/home/guest/Documents/Test1/ImageAI-master/imageai/Prediction/Weights/DenseNet.h5"))
prediction.loadModel()

pred_array = np.empty((0,6),dtype=object)
predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "1.jpg"), result_count=5 )
for img in os.listdir(TEST_PATH):
    if img.endswith('.jpg'):
        image = Image.open(os.path.join(TEST_PATH, img))
        image = image.convert("RGB")
        image = np.array(image, dtype=np.uint8)
        predictions, probabilities = prediction.predictImage(os.path.join(TEST_PATH, img), result_count=5)
        temprow = np.zeros((1,pred_array.shape[1]),dtype=object)
        temprow[0,0] = img
        for i in range(len(predictions)):
            temprow[0,i+1] = predictions[i]
        pred_array = np.append(pred_array,temprow,axis=0)


all_tags = pred_array[:,1:2].reshape(1,-1).tolist()
_in_sent = ' '.join(list(map(str,all_tags)))

#Storing image_name along with labels
mappings = []
for i in pred_array:
    for j in range(1):
        mappings.append([i[0],i[j+1]])

#Word2vec Model to generate vector values
model = Word2Vec(all_tags, min_count=1, size=2, alpha=0.025)
words = list(model.wv.vocab)
vector = list()
for word in words:
    vector.append(model[word])
X = model[model.wv.vocab]
model = Word2Vec(all_tags, min_count=1, size=2, alpha=0.025)
words = list(model.wv.vocab)
vector = list()
for word in words:
    vector.append(model[word])
    
#Storing image name
image_name = []
for i in range(len(vector)):
    image_name.append(mappings[i][0])

#Plotting the vector values in the plane
Xaxis = list()
yaxis = list()
myarray = np.reshape(vector[:(np.shape(vector)[0])], 2*(np.shape(vector)[0]))
for loop1 in range(2*(np.shape(vector)[0])):
    if(loop1%2 ==0):
        Xaxis.append(myarray[loop1])
    else:
        yaxis.append(myarray[loop1])
X = np.vstack((Xaxis, yaxis)).T


#plotting data
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

#for x, y in zip(X[:, 0], X[:, 1]):
#    plt.annotate(s='text', xy=(x, y), xytext=(-3, 3),
#        textcoords='offset points', ha='right', va='bottom')

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X, 'single')


plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
