import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
fd = mpFaceDetection.FaceDetection()
pTime = 0

while True:
    success,img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fd.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #print(detection.location_data.relative_bounding_box)
            #mpDraw.draw_detection(img, detection)
            ih, iw, ic = img.shape
            bboxc = detection.location_data.relative_bounding_box
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), \
                     int(bboxc.width * iw), int(bboxc.height * ih)
            cv2.rectangle(img, bbox, (255,0,255), 2)
            cv2.putText(img, f'Percent: {int(detection.score[0]*100)}', (bbox[0], bbox[1]-20 ), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0))

    cv2.imshow('Image', img)
    cv2.waitKey(1)

if __name__ == '__main__':
    main()
