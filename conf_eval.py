import numpy as np
import csv
import pandas as pd

def read_csv_to_2d_array(file_path):
    data = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # 各値を数値に変換できれば変換し、それ以外は文字列のままにする
            parsed_row = []
            for value in row:
                try:
                    # 浮動小数点数に変換可能なら変換
                    parsed_value = float(value) if '.' in value else int(value)
                except ValueError:
                    # 変換できない場合はそのまま文字列として扱う
                    parsed_value = value
                parsed_row.append(parsed_value)
            
            data.append(parsed_row)

    return data

# CSVファイルを読み込む
path = '/home/gpu-server-2/Desktop/ominato/fer2013/conf_result/confmat_all.csv'  # ここにCSVファイルのパスを指定
data = read_csv_to_2d_array(path)

# 各評価指標の配列を用意
tp  = np.zeros(7)
fp  = np.zeros(7)
fn  = np.zeros(7)
tn  = np.zeros(7)
pre = np.zeros(7)
rec = np.zeros(7)
f1  = np.zeros(7)
true_total = np.sum(data, axis=1)   # 各感情ラベルごとの枚数
pred_total = np.sum(data, axis=0)   # モデルが予測した各感情の枚数
total = np.sum(np.sum(data, axis=0))    # 全画像数

# 適合率(Precision), 再現率(Recall), F1-socreを計算
for i in range(7):
    tp[i] = data[i][i]
    fp[i] = pred_total[i] - tp[i]
    fn[i] = true_total[i] - tp[i]
    tn[i] = total - true_total[i] - pred_total[i] + tp[i]

    pre[i] = tp[i]/(tp[i]+fp[i])
    rec[i] = tp[i]/(tp[i]+fn[i])
    f1[i]  = 2*pre[i]*rec[i]/(pre[i]+rec[i])
acc = np.sum(tp)/total  # 正解率


# 計算された各評価指標を感情ラベルとともにcsvとして出力
label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprize', 'Neutral']
pre = np.round(pre, 3)
rec = np.round(rec, 3)
f1  = np.round(f1, 3)
out = np.vstack((label, pre, rec, f1))

df = pd.DataFrame(out)
df.to_csv('/home/gpu-server-2/Desktop/ominato/fer2013/conf_result/eval_all.csv', index=False, header=False)
print("Evaluation Complete")

print("Accuracy : {0}".format(acc))  # 正解率は表示