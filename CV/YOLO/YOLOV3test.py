import cv2
#joric ope
# Reading the Image
image = cv2.imread('../trials/pedestrian1.jpg')
print(image.shape)

#Configs:
#paths
LABEL_PATH='coco.names'
CONFIG_PATH='yolov3.cfg'
WEIGHT_PATH='yolov3.weights'
labels = open('coco.names').read().strip().split('\n')
print(labels)

yolo = cv2.dnn.readNetFromDarknet(CONFIG_PATH,WEIGHT_PATH)

layernames=yolo.getLayerNames()
print(yolo.getUnconnectedOutLayers())
layernames=[layernames[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
print(layernames)
blob = cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True, crop=False)
r = blob[0,0,:,:]   
r0=r.copy()
text=f'Blob shape={blob.shape}'
cv2.imshow('blob',r)
#cv2.displayOverlay('blob',text)
cv2.waitKey(1)

yolo.setInput(blob)
#4 pertama bounding boxnya (x,y,width,height)
#ke 5 box confidence
#80 sisanya class confidence
output=yolo.forward(layernames)
print(len(output))
#padding=(4, 4),   
# Drawing the regions in the Image

for item in output:
    print(item.shape)
for out in output:
    for gotcha in out:
        x, y, w, h= gotcha[0],gotcha[1],gotcha[2],gotcha[3]
        cv2.rectangle(r0, (int((x-w/2)*416), int((y-h/2)*416)), (int((x+w/2)*416), int((y+h/2)*416)), (100,100,100), 2)
  
# Showing the output Image
cv2.imshow("Image", r0)
cv2.waitKey(0)

cv2.destroyAllWindows()