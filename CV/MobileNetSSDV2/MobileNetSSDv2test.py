import cv2
import time
import numpy as np
# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco_2018_03_29.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
 
# Input image
img = cv2.imread('../trials/kelasft1.jpg')
rows, cols, channels = img.shape
classes = open('coco.names').read().strip().split('\n')
print(classes)
# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(320, 320), swapRB=True, crop=False))
 
# Runs a forward pass to compute the net output
t0 = time.time()
networkOutput = tensorflowNet.forward()
t = time.time()
print("time",t-t0)
print(networkOutput[0,0,0,0])
print(type(networkOutput))
print(networkOutput.shape)
print(networkOutput[0].shape)
print("tesshape")
print(networkOutput[0].shape)
# Loop on the outputs
for detection in networkOutput[0,0]:
    
    score = float(detection[2])
    if score > 0.2 and int(detection[1])==1:
     
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
 
        #draw a red rectangle around detected objects
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
        cv2.putText(img,"confidence:"+str(round(score,3)),(int(left),int(top)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
 
# Show the image with a rectagle surrounding the detected objects 
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()