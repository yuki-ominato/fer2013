import cv2
import os

# Haar Cascade分類器のXMLファイルを読み込む
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 画像が保存されているディレクトリのパス
image_folder = '/home/gpu-server-2/Desktop/ominato/fer2013/FER2013_data/all_train'
output_folder = '/home/gpu-server-2/Desktop/ominato/fer2013/FER2013_data/non_face'

os.makedirs(output_folder, exist_ok=True)

face = 0
non_face = 0

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {filename}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            non_face += 1
        else:
            face += 1

        # 明示的に変数を解放
        del img, gray, faces
        cv2.destroyAllWindows()

print(face)
print(non_face)
