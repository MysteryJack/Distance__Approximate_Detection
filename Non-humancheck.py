import numpy as np
import pandas as pd
import os

box = pd.read_csv("C:/Users/JackJSC/Desktop/Coding/AI_builder/HumanDetector/challenge-2019-validation-detection-bbox.csv")
human_d = pd.read_csv("C:/Users/JackJSC/Desktop/Coding/AI_builder/HumanDetector/challenge-2019-validation-detection-human-imagelabels.csv")
ibox = box.index
not_human = []
for i in human_d.index:
    if human_d["Confidence"][i] == 0:
        not_human.append(human_d['ImageID'][i])
for y in not_human:
    ibox = box.index
    for x in ibox[box["ImageID"] == y]:
        box.drop(x,axis=0,inplace=True)        


box.to_csv("C:/Users/JackJSC/Desktop/Coding/AI_builder/HumanDetector/output/challenge-2019-validation-detection-bbox.csv")
human_d.to_csv("C:/Users/JackJSC/Desktop/Coding/AI_builder/HumanDetector/output/challenge-2019-validation-detection-human-imagelabels.csv")

for y in not_human:
    inhuman = "C:/Users/JackJSC/Desktop/Coding/AI_builder/HumanDetector/validation/" + y + ".jpg"
    try:
        os.remove(inhuman)
    except:
        pass