from imageai.Prediction import ImagePrediction
import os
import pandas as pd
from PIL import Image
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import json
import statistics


execution_path = os.getcwd()
test_path = '/home/guest/Documents/Aikomi'

prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "/home/guest/Documents/Test1/ImageAI-master/imageai/Prediction/Weights/DenseNet.h5"))
prediction.loadModel()

#To return the 5tags Json table
def model_calc(test_path):
    pred_array = np.empty((0,6),dtype=object)
    for img in os.listdir(test_path):
        if img.endswith('.jpg'):
            image = Image.open(os.path.join(test_path, img))
            image = image.convert("RGB")
            image = np.array(image, dtype=np.uint8)
            predictions, probabilities = prediction.predictImage(os.path.join(test_path, img), result_count=5)
            temprow = np.zeros((1,pred_array.shape[1]),dtype=object)
            temprow[0,0] = img
            for i in range(len(predictions)):
                temprow[0,i+1] = predictions[i]
            pred_array = np.append(pred_array,temprow,axis=0)

    tag_df = pd.DataFrame(data=pred_array,columns=['File_name','Tag_1','Tag_2','Tag_3','Tag_4','Tag_5'])  
#    tag_df.to_json(r'tag_df_json_index.json',orient='table')
#    tag_df_json_index = json.load(open('tag_df_json_index.json'))
    return tag_df


#to return the image_name and corresponding top_tag
def img_map(tag_df):
    all_tags = tag_df[:,1:2].reshape(1,-1).tolist()
    _in_sent = ' '.join(list(map(str,all_tags)))
    mappings = []
    for i in tag_df:
        for j in range(1):
            mappings.append([i[0],i[j+1]])
    return mappings



#Word2vec model
def w2v(all_tags):
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
    return vector

#fetching the image_name in one list
def image(vector):
    image_name = []
    for i in range(len(vector)):
        image_name.append(mappings[i][0])
    return image_name



#Plotting the vector values in the plane and returning co-ordinates
def co_or(vector):
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


#KMeans clustering and returning scatterplot
def K_Means():
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    x = np.random.rand(15)
    y = np.random.rand(15)
    names = np.array(list(image_name))
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
    return plt.show()



