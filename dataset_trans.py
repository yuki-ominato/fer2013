"""
This is New preprocessing.py

CHECK LINE 75,76
"""
import pandas as pd
import numpy as np
import cv2

NUM = 2000 # 感情ごとの生成枚数
data_path_angry      = "./gen_images/angry/"     # Label 0
data_path_disgust    = "./gen_images/disgust/"   # Label 1
data_path_fear       = "./gen_images/fear/"      # Label 2
data_path_happy      = "./gen_images/happy/"     # Label 3
data_path_sad        = "./gen_images/sad/"       # Label 4
data_path_surprise   = "./gen_images/surprise/"  # Label 5
data_path_neutral    = "./gen_images/neutral/"   # Label 6

data = pd.read_csv('./fer2013.csv') # fer2013 dataset PATH num:35887
# print(data)

width, height = 48, 48

datapoints = data["pixels"].tolist()
# print(np.array(datapoints).shape)

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

using_data = [data_path_angry,
              data_path_disgust,
              data_path_fear,
              data_path_happy,
              data_path_sad,
              data_path_surprise,
              data_path_neutral]

for Data_path in using_data:
    for i in range(NUM):
        img = cv2.imread(Data_path+"seed{0:04d}.png".format(i))
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_g_rs = cv2.resize(img_g,(48,48),interpolation=cv2.INTER_AREA)
        X.append(img_g_rs.astype("float32"))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
# y = pd.get_dummies(data['emotion']).as_matrix()
y = pd.get_dummies(data['emotion']).values
label_angry    = np.array([[1,0,0,0,0,0,0]])
label_disgust  = np.array([[0,1,0,0,0,0,0]])
label_fear     = np.array([[0,0,1,0,0,0,0]])
label_happy    = np.array([[0,0,0,1,0,0,0]])
label_sad      = np.array([[0,0,0,0,1,0,0]])
label_surprise = np.array([[0,0,0,0,0,1,0]])
label_neutral  = np.array([[0,0,0,0,0,0,1]])

using_label = [label_angry,
               label_disgust,
               label_fear,
               label_happy,
               label_sad,
               label_surprise,
               label_neutral]

for Label in using_label:    
    for i in range(NUM):
        y = np.append(y, Label, axis=0)

print(X.shape)
print(y.shape)

##### IMPORTANT ALWAYS CHANGE #####
X_name = "fdataX_all"
y_name = "flabels_all"

#storing them using numpy
np.save('./fdataX/' + X_name, X)
np.save('./flabels/' + y_name, y)