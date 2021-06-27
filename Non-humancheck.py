import numpy as np
import pandas as pd
import os

#path to human parts detecting label
boxy = pd.read_csv("/path/label.csv")
#path to picture
pic_folder = os.fsencode("path/to/picture/picture.jpeg")
#path to human detecting label
human_d = pd.read_csv("/path/human_detection_label.csv")
ibox = box.index
not_human = []
for i in human_d.index:
    if human_d["Confidence"][i] == 0:
        not_human.append(human_d['ImageID'][i])
for y in not_human:
    ibox = box.index
    for x in ibox[box["ImageID"] == y]:
        box.drop(x,axis=0,inplace=True)        


box.to_csv("/path/label.csv")
human_d.to_csv("/path/human_detection_label.csv")

for y in not_human:
    inhuman = "/path/to/picture/" + y + ".jpg"
    try:
        os.remove(inhuman)
    except:
        pass
