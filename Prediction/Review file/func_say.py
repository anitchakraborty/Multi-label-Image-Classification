from imageai.Prediction import ImagePrediction
import os
import pandas as pd
import numpy as np
from PIL import Image
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

#execution_path = os.getcwd()
#prediction = ImagePrediction()
#prediction.setModelTypeAsDenseNet()
#prediction.setModelPath(os.path.join(execution_path, "/home/guest/Documents/Test1/ImageAI-master/imageai/Prediction/Weights/DenseNet.h5"))
#prediction.loadModel()
#TEST_PATH = '/home/guest/Documents/Aikomi'



def Den():
    execution_path = os.getcwd()
    prediction = ImagePrediction()
    prediction.setModelTypeAsDenseNet()
    prediction.setModelPath(os.path.join(execution_path, "/home/guest/Documents/Test1/ImageAI-master/imageai/Prediction/Weights/DenseNet.h5"))
    prediction.loadModel()
    return prediction


def Res():
    execution_path = os.getcwd()
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(execution_path, "/home/guest/Documents/Test1/ImageAI-master/imageai/Prediction/Weights/ResNet.h5"))
    prediction.loadModel()
    return prediction




    
def tag(TEST_PATH, prediction):
    pred_array = np.empty((0,6),dtype=object)
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
            return pred_array

pred_array = tag(TEST_PATH)


def mapping(pred_array):
        all_tags = pred_array[:,1:6].reshape(1,-1).tolist()
        _in_sent = ' '.join(list(map(str,all_tags)))
        mappings = []
        for i in pred_array:
            for j in range(1):
                mappings.append([i[0],i[j+1]])
                return mappings, all_tags

####### Calling only one return position wise//if you need both, usual call
mappings,_ = mapping(pred_array)
_,all_tags= mapping(pred_array)
                

def w2v(all_tags):
    model = Word2Vec(all_tags, min_count=1, size=2, alpha=0.025)
    words = list(model.wv.vocab)
    vector = list()
    for word in words:
        vector.append(model[word])
    model = Word2Vec(all_tags, min_count=1, size=2, alpha=0.025)
    words = list(model.wv.vocab)
    vector = list()
    for word in words:
        vector.append(model[word])
    return vector, words


vector,words=w2v(all_tags)
def imglst(vector, mappings):
    image_name = []
    for i in range(len(vector)):
        image_name.append(mappings[i][0])
        return image_name
        


def co_ord(vector):
    Xaxis = list()
    yaxis = list()
    myarray = np.reshape(vector[:(np.shape(vector)[0])], 2*(np.shape(vector)[0]))
    for loop1 in range(2*(np.shape(vector)[0])):
        if(loop1%2 ==0):
            Xaxis.append(myarray[loop1])
        else:
            yaxis.append(myarray[loop1])
    X = np.vstack((Xaxis, yaxis)).T
    return X


image_name=w2v(all_tags)
X=co_ord(vector)
def K_mean(X, image_name):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    c = np.random.randint(1,5,size=522)
    
    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn
    
    fig,ax = plt.subplots()
    sc = plt.scatter(X[:,0], X[:,1],c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                               " ".join([image_name[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)
    
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", hover)
    return y_kmeans

##fetching the plt.show() by calling K_mean(X, image_name)
#y_kmeans=K_mean(X, image_name)
#plt.show()


y_kmeans=K_mean(X, image_name)
def clustdf(y_kmeans, X, words, image_name):
    cluster_center = list()
    for center in range(len(X)):
        cluster_center.append(y_kmeans[center])
    tag_cluster = pd.DataFrame({'cluster_center': cluster_center,'image_tags': words })
    mappings = {k:v for k,v in zip(words, image_name)}
    tag_cluster['image_name'] = pd.Series([mappings[i] for i in tag_cluster.image_tags])
    #df_cluster1 = tag_cluster[tag_cluster['cluster_center']==0]
    #df_cluster2 = tag_cluster[tag_cluster['cluster_center']==1]
    #df_cluster3 = tag_cluster[tag_cluster['cluster_center']==2]
    #df_cluster4 = tag_cluster[tag_cluster['cluster_center']==3]
    return tag_cluster


def main(model_choice = 'Res'):
    TEST_PATH = '/home/guest/Documents/Aikomi'
    if model_choice=='Res':
        prediction = Res()
    elif model_choice=='Den':
        prediction = Den()
    tag(TEST_PATH)
    pred_array = tag(TEST_PATH, prediction)
    mappings,all_tags = mapping(pred_array)
    vector,words=w2v(all_tags)
    image_name=w2v(all_tags)
    X=co_ord(vector)
    y_kmeans=K_mean(X, image_name)
    #plt.show()
    tag_cluster=clustdf(y_kmeans, X, words, image_name)
    print(tag_cluster)

if __name__=="__main__":
    main(model_choice)
    