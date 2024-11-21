import shutil
import os

inputpath = "/home/gpu-server-2/Desktop/ominato/fer2013/FER2013_data/train"
outputpath = "/home/gpu-server-2/Desktop/ominato/fer2013/FER2013_data/all_train"

# ディククトリ作成
os.makedirs(outputpath, exist_ok=True)

# label_folderにinputpath内のfolderを格納
for label_folder in os.listdir(inputpath):
    # inputpathとlabel_folderを繋げ，パスにする
    label_path = os.path.join(inputpath, label_folder)

    if(os.path.isdir(label_path)):
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)

            new_image_name = f"{label_folder}_{image_file}"
            new_image_path = os.path.join(outputpath, new_image_name)

            # 画像のコピー
            shutil.copy(image_path, new_image_path)
print(f"All images have been combined into the directory: {outputpath}")