"""
CHECK LINE 13, 18, 23, 24, 42, 43
"""
# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import numpy as np

json_file = open('./json/fer_all.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./h5/fer_all.h5")
print("Loaded model from disk")

truey=[]
predy=[]
x = np.load('./modXtest/modXtest_all.npy')
y = np.load('./modytest/modytest_all.npy')

yhat= loaded_model.predict(x)
yh = yhat.tolist()
yt = y.tolist()
count = 0

for i in range(len(y)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1

acc = (count/len(y))*100

#saving values for confusion matrix and analysis
np.save('./truey/truey_all', truey)
np.save('./predy/predy_all', predy)
print("Predicted and true label values saved")
print("Accuracy on test set :"+str(acc)+"%")
