import cv2
import numpy as np
import csv

f = open('./augmentation_data/angry.csv', 'w', newline='')
data =[]
outdir = "/home/gpu-server-2/Desktop/ominato/fer2013/model_shanks/augmentation_data"
for i in range(10):
  img = cv2.imread("./gen_images/seed{0:04d}.png".format(i))
  img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_g_rs = cv2.resize(img_g,(48,48),interpolation=cv2.INTER_AREA)
  d = np.asarray(img_g_rs).reshape(48*48, -1)
  data.append(d.astype("float32"))

print(img.shape)
print(img_g_rs.shape)
print(np.array(data).shape)

writer = csv.writer(f)
writer.writerows(data)
f.close()
