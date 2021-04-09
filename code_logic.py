import time
import math
import cv2
import numpy as np

confid = 0.5
thresh = 0.5
# vname=""
# vname=input("Video name in videos folder:  ")
# if(vname==""):
vname = ""
vid_path = "sample2.mp4"  # index.jpg

# Calibration needed for each video


labelsPath = "./YoloFiles/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

weightsPath = "./YoloFiles/yolov3.weights"
configPath = "./YoloFiles/yolov3.cfg"

###### use this for faster processing (caution: slighly lower accuracy) ###########

# weightsPath = "./yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
# configPath = "./yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)  #to load model
ln = net.getLayerNames()  #to get layers
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
FR = 0
vs = cv2.VideoCapture(vid_path)

# vs = cv2.imread(vid_path)
# vs = cv2.VideoCapture(0)  ## USe this if you want to use webcam feed
frame = vs


(W, H) = (None, None)
frame
fl = 0
q = 0
count = 0
while True:

    (grabbed, frame) = vs.read()

    (H, W) = frame.shape[:2]
    # print(H,W)
    cv2.line(frame, (0, H - 100), (W, H - 100), (66, 66, 255), 3)

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:

        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if LABELS[classID] == "truck" or "car" or "bus" or "motorbike":

                if confidence > confid:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    if len(idxs) > 0:

        idf = idxs.flatten()

        # emptlst = [0]
        for i in idf:

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if cen[1] >= H - 100:
                count = count + 1

        cv2.putText(frame, "Total Vehicle : = " + str(count), (W-400, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 3)
        frame = cv2.resize(frame, (900, 600),
                           interpolation=cv2.INTER_AREA)

        cv2.imshow('Vehicle Counter', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break






cv2.destroyAllWindows()

# if writer is None:
#    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#    writer = cv2.VideoWriter("op_"+vname, fourcc, 30,
#                             (frame.shape[1], frame.shape[0]), True)

# writer.write(frame)
# print("Processing finished: open"+"op_"+vname)
# writer.release()
#                                                                                                                   vs.release()
