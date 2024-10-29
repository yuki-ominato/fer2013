import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import csv

# CHECK LINE 9, 10, 37
#test

y_true = np.load('./truey/truey_all.npy')
y_pred = np.load('./predy/predy_all.npy')
cm = confusion_matrix(y_true, y_pred)
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
title='Confusion matrix'
print(cm)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()



# csvファイルに保存
file_path = '/home/gpu-server-2/Desktop/ominato/fer2013/conf_result/confmat_all.csv'
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(cm)

print("CSVファイルに保存されました。")