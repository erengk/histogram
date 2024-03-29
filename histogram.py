import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

#Dataları okuyup fazla boşlukları silen fonksiyon:
def read_data_and_strip(file_path, seperator):
    file = pd.read_csv(file_path, sep=seperator)
    file.pop(file.columns[0])
    file.pop(file.columns[-1])
    file.columns = file.columns.str.strip()
    return file

#Dataların yollarını verip okuduğum kodlar
path = pathlib.Path(__file__).parent.resolve()
path = str(path)

file1 = read_data_and_strip(file_path= path+"/Event-scan-1.dat", seperator='*')
file2 = read_data_and_strip(file_path= path+"/Event-scan-2.dat",seperator='*')
#2 farklı data setini column ismine göre birleştirdim
merged_data = pd.merge(file1, file2, on='Row')

print(merged_data)

plt.plot(merged_data["EventNo"], merged_data["PMT_Signa"], color="g")
plt.title("Event No - PMT Signal Count Grafiği")
plt.xlabel("Event No")
plt.ylabel("PMT Signal Count")
plt.legend(["f(Event_No)=PMT_Signal_Count"])
plt.show