import cv2
import mediapipe as mp
import face_mesh
from face_mesh import FaceMeshDetector
from Utils import putTextRect,stackImages
import numpy as np

 
cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
 
textList = ["The only" ,"way to" ,"learn a new", "programming language", "is by", "writing programs " "in it." ,"- Dennis Ritchie."]
 
sen = 50  # more is less
 
while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)
 
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3
 
        # Finding distance
        f = 840
        d = (W * f) / w
        print(d)
 
        putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)
 
        for i, text in enumerate(textList):
            singleHeight = 20 + int((int(d/sen)*sen)/4)
            scale = 0.4 + (int(d/sen)*sen)/75
            cv2.putText(imgText, text, (50, 50 + (i * singleHeight)),
                        cv2.FONT_ITALIC, scale, (255, 255, 255), 2)
 
    imgStacked = stackImages([img, imgText], 2, 1)
    cv2.imshow("Image", imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
