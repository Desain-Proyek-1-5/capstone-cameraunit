##https://www.youtube.com/watch?v=cdblJqEUDNo
##USAGE
#yolov3_video.py -a 1 -c 0.2 -i ../trials/pltsft.jpg

# YOLO object detection
import cv2 as cv
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-a','--algorithm', type=int , metavar='', required=True, help='Algoritma yang akan digunakan')
parser.add_argument('-c', '--confidence', type=float, metavar='', required=True, help='Nilai confidence yang digunakan')
##parser.add_argument("-i", "--image", required=True, default="video_kelas.mp4",
#help="path to input image")
args = parser.parse_args()

treshold = args.confidence

algorithm = args.algorithm

cap = cv.VideoCapture("video_kelas.mp4")

if algorithm == 1 :
    algorithm_cfg = 'yolov3.cfg'
    algorithm_weight = 'yolov3.weights'
elif algorithm == 2 :
    algorithm_cfg = 'yolov3-tiny.cfg'
    algorithm_weight = 'yolov3-tiny.weights'

def main():
    while True:
        ret, frame = cap.read()

        # Load names of classes and get random colors
        classes = open('coco.names').read().strip().split('\n')
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

        # Give the configuration and weight files for the model and load the network.
        net = cv.dnn.readNetFromDarknet(algorithm_cfg, algorithm_weight)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        # determine the output layer
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the image
        blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]

        #cv.imshow('blob', r)
        text = f'Blob shape={blob.shape}'
        #cv.displayOverlay('blob', text)
        cv.waitKey(1)

        net.setInput(blob)
        t0 = time.time()
        outputs = net.forward(ln)
        t = time.time()
        print('time=', t-t0)

        print(len(outputs))
        for out in outputs:
            print(out.shape)

        def trackbar2(x):
            confidence = x/100
            r = r0.copy()
            for output in np.vstack(outputs):
                if output[4] > confidence:
                    x, y, w, h = output[:4]
                    p0 = int((x-w/2)*416), int((y-h/2)*416)
                    p1 = int((x+w/2)*416), int((y+h/2)*416)
                    cv.rectangle(r, p0, p1, 1, 1)
            #cv.imshow('blob', r)
            text = f'Bbox confidence={confidence}'
            #cv.displayOverlay('blob', text)

        r0 = blob[0, 0, :, :]
        r = r0.copy()
        ##cv.imshow('blob', r)
        ##cv.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
        trackbar2(50)

        boxes = []
        confidences = []
        classIDs = []
        h, w = frame.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > treshold:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv.dnn.NMSBoxes(boxes, confidences, treshold, 0.5)
        print("detections:",len(confidences))
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv.putText(frame, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv.imshow('frame',frame)
        #cv2.imshow('gray',gray)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
 
if __name__== '__main__' :
    main()
