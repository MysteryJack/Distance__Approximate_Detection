import numpy as np
import tensorflow as tf
import cv2
import time


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


model_path = r"C:\Users\JackJSC\Desktop\Coding\AI builder\HumanDetector\faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.8
line_op = 0.5
cap = cv2.VideoCapture(0)
img_array = []
start_time = time.time()

while True:
    r, img = cap.read()
    if r:  
      size = (1280,720) 
      try:
        img = cv2.resize(img, size)
      except:
        print("Error Resizing")
      line_layer = np.zeros(img.shape, np.uint8)
      box_layer = np.zeros(img.shape, np.uint8)
      boxes, scores, classes, num = odapi.processFrame(img)
      center = []
      distance = []
        
      # Visualization of the results of a detection.
      for i in range(len(boxes)):
          # Class 1 represents human
          if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            b_center = (int(box[3]/2+box[1]/2), int(box[2]/2+box[0]/2))
            center.append(b_center)
            cv2.rectangle(box_layer,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
            cv2.putText(box_layer, 'Person ' + str(len(center)), (box[1],box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
      nbox = len(center)
    
      for n in range(nbox) :
          for m in [i + (n + 1) for i in range((nbox-1)-n)]:
              cv2.line(line_layer, center[n], center[m], (255,255,255),2)
        
      o1_layer = cv2.addWeighted(img, 1.0, line_layer, line_op, 0)
      o2_layer = cv2.addWeighted(o1_layer, 1.0, box_layer, 1.0, 0)
        

      img_array.append(o2_layer)
      out = cv2.VideoWriter('o_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
      for i in range(len(img_array)):
          out.write(img_array[i])
    elif time.time() - start_time == 10:
      end_time = time.time()
      out.release()
      print("Total Elapsed Time:", end_time-start_time)
      break
cap.release()
