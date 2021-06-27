import numpy as np
import tensorflow as tf
import cv2
import time
import pandas as pd
import os

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)
        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


model_path = "/path/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.8
line_op = 0.5
img_array = []
start_time = time.time()
#id of human parts detection(head,leg,arm)
human_parts = ["/m/04hgtk", "/m/035r7c", "/m/0dzf4"]
#path to label
boxy = pd.read_csv("/path/label.csv")
#path to picture
pic_folder = os.fsencode("path/to/picture/picture.jpeg")
total_acc = []

for img_file in os.listdir(pic_folder):
    pic_name = os.fsdecode(img_file)
    pic_code = pic_name[:-4]
    img = cv2.imread("path/to/picture/" + pic_name)
    size = (1280,720) 
    relavent = boxy[boxy["ImageID"]==pic_code]
    relavent = relavent[boxy["LabelName"]=="/m/04hgtk"].combine_first(relavent[boxy["LabelName"]=="/m/035r7c"].combine_first(relavent[boxy["LabelName"]=="/m/0dzf4"]))
    accuracy = []
    print(pic_code)
    try:
        img = cv2.resize(img, size)
    except:
        print("Error Resizing")
    box_layer = np.zeros(img.shape, np.uint8)
    boxlist, scores, classeslist, num = odapi.processFrame(img)
    boxes = pd.Series(boxlist)
    classes = pd.Series(classeslist)
    #output true if has something
    h_or_check = ((boxy["ImageID"][boxy["LabelName"]=="/m/04hgtk"]==pic_code).any(axis = 0) or 
            (boxy["ImageID"][boxy["LabelName"]=="/m/035r7c"]==pic_code).any(axis = 0) or
            (boxy["ImageID"][boxy["LabelName"]=="/m/0dzf4"]==pic_code).any(axis = 0)
            )
    #if model cant detected and there is really nothing
    if boxes[classes == 1].to_list()[0] == (0,0,0,0)  and not h_or_check:
        total_acc.append(1)
        print("Nothing")
        print("Accuracy: " + str(sum(total_acc)/len(total_acc)))
        continue
    
    #if model detect something but there is nothing
    if boxes[classes == 1].to_list()[0] == (0,0,0,0) and not h_or_check:
        total_acc.append(0)
        print("Fake")
        print("Accuracy: " + str(sum(total_acc)/len(total_acc)))
        continue

    #if model detect nothing but there is something
    if boxes[classes == 1].to_list()[0] == (0,0,0,0) and h_or_check:
        total_acc.append(0)
        print("Ghost")
        print("Accuracy: " + str(sum(total_acc)/len(total_acc)))
        continue
    for i in range(len(boxes)):
        acc = 0
        for x in range(len(relavent)):
        # Class 1 represents human
          if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            
            inter_area = ((min(box[3], 1280 * (relavent["XMax"].to_list()[x])) - max(box[1], 1280 * (relavent["XMin"].to_list()[x])))) * ((min(box[2], 720* (relavent["YMax"].to_list()[x])) - max(box[0], 720* (relavent["YMin"].to_list()[x]))))
            try:
                acc += inter_area/(((1280 * (relavent["XMax"].to_list()[x]))-(1280 * (relavent["XMin"].to_list()[x])))*((720* (relavent["YMax"].to_list()[x]))-(720* (relavent["YMin"].to_list()[x]))))
            except:
                continue
        if acc > 0: accuracy.append(1) 
        else: accuracy.append(0)
    total_acc.append(sum(accuracy)/len(accuracy))
    print("Accuracy: " + str(sum(total_acc)/len(total_acc)))
end_time = time.time()
print("Total Elapsed Time:", end_time-start_time)

