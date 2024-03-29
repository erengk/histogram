import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from numpy import ravel
import sklearn.metrics as mt
from xgboost import XGBClassifier
import time
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

def get_headers(data_frame):
    return list(data_frame.columns.values)

#Dataları okuyup fazla boşlukları silen fonksiyon:
def read_data_and_strip(file_path, seperator):
    file = pd.read_csv(file_path, sep=seperator)
    file.pop(file.columns[0])
    file.pop(file.columns[-1])
    file.columns = file.columns.str.strip()
    return file


def encoder(string_array):
    my_new_encoded_array = []
    for key in string_array:
        if key == "IBD":
            my_new_encoded_array.append(0)
        elif key == "Fon":
            my_new_encoded_array.append(1)
    return my_new_encoded_array

def decoder(numeric_array):
    my_new_decoded_array = []
    for key in numeric_array:
        if key == 0:
            my_new_decoded_array.append("IBD")
        elif key == 1:
            my_new_decoded_array.append("Fon")
    return my_new_decoded_array


#============================================MAIN=============================================#
path = pathlib.Path(__file__).parent.resolve()
path = str(path)
#IBD dosyalarını al:
ibd_path = path+"/IBD/"
event_scan_1 = read_data_and_strip(file_path=ibd_path+"Event-scan-1.dat", seperator='*')
event_scan_2 = read_data_and_strip(file_path=ibd_path+"Event-scan-2.dat",seperator='*')
pmt_scan = read_data_and_strip(file_path=ibd_path+"PMT-scan.dat",seperator='*')

#IBD datalarını merge et
event_scan = pd.merge(event_scan_2,event_scan_1,  on='Row')
merged_ibd_data = pd.merge(event_scan, pmt_scan, on='EventNo')
merged_ibd_data["label"] = 'IBD'

#Fon PLANE dosyalarını al:
fon_plane_path = path+"/Fon/plane/"
event_scan_plane = read_data_and_strip(file_path=fon_plane_path+"Event-scan.dat", seperator='*')
gun_scan = read_data_and_strip(file_path=fon_plane_path+"Gun-scan.dat",seperator='*')
pmt_scan = read_data_and_strip(file_path=fon_plane_path+"PMT-scan.dat",seperator='*')

#Fon PLANE datalarını merge et
event_scan = pd.merge(event_scan_plane, gun_scan, on='EventNo')
merged_fon_plane_data = pd.merge(event_scan, pmt_scan, on='EventNo')
#merged_fon_plane_data["label"] = 'Fon_Plane'

#Fon SPHERE dosyalarını al:
fon_sphere_path = path+"/Fon/sphere/"
event_scan = read_data_and_strip(file_path=fon_sphere_path+"Event-scan.dat", seperator='*')
gun_scan = read_data_and_strip(file_path=fon_sphere_path+"Gun-scan.dat",seperator='*')
pmt_scan = read_data_and_strip(file_path=fon_sphere_path+"PMT-scan.dat",seperator='*')

#Fon SPHERE datalarını merge et
event_scan = pd.merge(event_scan, gun_scan, on='EventNo')
merged_fon_sphere_data = pd.merge(event_scan, pmt_scan, on='EventNo')
#merged_fon_sphere_data["label"] = 'Fon_Sphere'

#Dataları Alt Alta Birleştir:
merged_fon_data = pd.merge(merged_fon_sphere_data, merged_fon_plane_data, on='EventNo')
merged_fon_data["label"] = 'Fon'
merged_all_data = pd.concat([merged_fon_data,merged_ibd_data])

#nan değerlerini modele uygun -1 değeri ata:
merged_all_data.replace("      -nan ",-1, inplace=True)
merged_all_data = merged_all_data.fillna(-1)

#featurelar
ibd_features = event_scan_2[["PMT_Signa","PMT_Promp","PMT_Delay","PMT_Total","PMT_TotalA"]]
ibd_features = ibd_features.assign(label='IBD')
ibd_features_length = len(ibd_features)
plane_features = event_scan_plane[["PMT_Signa","PMT_Promp","PMT_Delay","PMT_Total","PMT_TotalA"]]
plane_features = plane_features.assign(label='Fon')
plane_features_length = len(plane_features)
merged_features = pd.concat([ibd_features,plane_features])
print(merged_features)


#->XGBoost MODEL EĞİTİMİ:
accuracyResault = []
for i in range(1,10):
    testSize = i*0.1
    X_train, X_test, Y_train, Y_test = train_test_split(merged_features.drop('label', axis=1), merged_features["label"], random_state=42, test_size=testSize)
    Y_train = ravel(Y_train)
    Y_test = ravel(Y_test)
    xgboost_model = XGBClassifier()
    numeric_y_train = encoder(Y_train)
    xgboost_model.fit(X_train, numeric_y_train)
    numeric_y_pred = xgboost_model.predict(X_test)
    y_pred = decoder(numeric_y_pred)
    accuracy = accuracy_score(Y_test, y_pred)
    accuracyResault.append([i,accuracy*100.0])

#Modeli Test Et:
score = mt.f1_score(Y_test, y_pred, average='weighted')
print("Prediction Score:")
print(score)


#Accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#Grafikleştirme

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = [str(testSize)+" Test",
          str(1-testSize)+" Eğitim",
          ]

data = [float(x.split()[0]) for x in recipe]
ingredients = [x.split()[-1] for x in recipe]


def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n"


wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients,
          title="Renk Tablosu",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("Veri Seti Eğitim ve Test Oranı")

plt.show()

def f(x):
    return accuracyResault[x-1][1]

x_degerleri = list(range(1,10))
y_degerleri = [f(x) for x in x_degerleri]
plt.plot(x_degerleri, y_degerleri, color="g")
plt.title("Accuracy ve Test Size Grafiği")
plt.xlabel("Test Size")
plt.ylabel("Accuracy")
plt.show()