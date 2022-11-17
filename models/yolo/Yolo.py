import cv2
import numpy as np
import torch

from util import utils


class YoloV3:

    CLASS_PERSON = 0
    CLASS_BALL = 32

    def __init__(self):
        self.labels_file = f"{utils.get_yolov3_model_path()}yolov3.txt"
        self.config_file = f"{utils.get_yolov3_model_path()}yolov3.cfg"
        self.weights_file = f"{utils.get_yolov3_model_path()}yolov3.weights"
        self.desired_conf = .0
        self.desired_thres = .3
        self.frame = None

        # print("[INFO] Loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(self.config_file, self.weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        layer_names = self.net.getLayerNames()
        self.layer_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, image):
        self.frame = image
        blob = cv2.dnn.blobFromImage(self.frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        return self.net.forward(self.layer_names)

    def extract_objects(self, layer_outputs):

        h, w = self.frame.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        for output in layer_outputs:

            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.desired_conf and classID in (self.CLASS_PERSON, self.CLASS_BALL):

                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    if x < 0 or y < 0 or width / self.frame.shape[0] > .5 or height / self.frame.shape[1] > .5:
                        continue

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # return array of elements
        # (bounding_box, pred_confidence, pred_class, valid_box)
        # where valid box => person or soccer ball (to be refined later)
        return [[box, conf, cls, None] for (box, conf, cls) in zip(boxes, confidences, classIDs)]

    def merge_overlapping_boxes(self, objs):
        box_list = [box for (box, _, _, _) in objs]
        conf_list = [conf for (_, conf, _, _) in objs]
        ids = cv2.dnn.NMSBoxes(box_list, conf_list, self.desired_conf, self.desired_thres)
        if ids.__len__() == objs.__len__():
            return objs
        ids = ids.flatten()
        return [objs[id] for id in ids]


class YoloV5:

    def __init__(self):
        weights_file = f'{utils.get_yolov5_model_path()}yolov5s'
        self.net = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def predict(self, image):
        return self.net(image)

    def extract_objects(self, layer_outputs):
        print('gr')


if __name__ == '__main__':

    model = YoloV5()
    image_file = f'{utils.get_project_root()}demo/frame.jpg'
    image = cv2.imread(image_file)
    preds = model.predict(image)
    print('gr')
