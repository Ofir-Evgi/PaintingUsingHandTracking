import cv2
import mediapipe as mp
import time

videoStream = cv2.VideoCapture(0)

mediapipeHands = mp.solutions.hands
handsModule = mediapipeHands.Hands()
drawingUtils = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    readSuccess, videoFrame = videoStream.read()
    frameRGB = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2RGB)
    handResults = handsModule.process(frameRGB)

    if handResults.multi_hand_landmarks:
        for singleHand in handResults.multi_hand_landmarks:
            for markerID, landmark in enumerate(singleHand.landmark):
                frameHeight, frameWidth, _ = videoFrame.shape
                landmarkX, landmarkY = int(landmark.x * frameWidth), int(landmark.y * frameHeight)
                print(markerID, landmarkX, landmarkY)

                cv2.circle(videoFrame, (landmarkX, landmarkY), 8, (255, 0, 255), cv2.FILLED)

            drawingUtils.draw_landmarks(videoFrame, singleHand, mediapipeHands.HAND_CONNECTIONS)

    currentTime = time.time()
    framesPerSecond = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(videoFrame, str(int(framesPerSecond)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", videoFrame)
    cv2.waitKey(1)
