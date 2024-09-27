import cv2
import mediapipe as mp
import time

class HandTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.handsInstance = self.mpHands.Hands(static_image_mode=self.mode,
                                                max_num_hands=self.maxHands,
                                                min_detection_confidence=self.detectionCon,
                                                min_tracking_confidence=self.trackCon)
        self.drawingUtils = mp.solutions.drawing_utils
        self.fingertipIds = [4, 8, 12, 16, 20]

    def findHands(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.handsInstance.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.drawingUtils.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, handNo=0, draw=True):
        self.landmarkList = []

        if self.results.multi_hand_landmarks:
            targetHand = self.results.multi_hand_landmarks[handNo]
            for markerIndex, landmark in enumerate(targetHand.landmark):
                height, width, _ = image.shape
                coordX, coordY = int(landmark.x * width), int(landmark.y * height)
                self.landmarkList.append([markerIndex, coordX, coordY])
                if draw:
                    cv2.circle(image, (coordX, coordY), 8, (255, 0, 255), cv2.FILLED)

        return self.landmarkList

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.landmarkList[self.fingertipIds[0]][1] < self.landmarkList[self.fingertipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for fingertipID in range(1, 5):
            if self.landmarkList[self.fingertipIds[fingertipID]][2] < self.landmarkList[self.fingertipIds[fingertipID] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    prevTime = 0
    currTime = 0
    videoCapture = cv2.VideoCapture(0)
    handTrackerInstance = HandTracker()
    while True:
        frameSuccess, videoFrame = videoCapture.read()
        videoFrame = handTrackerInstance.findHands(videoFrame)
        landmarks = handTrackerInstance.findPosition(videoFrame)

        if len(landmarks) != 0:
            print(landmarks[4])

        currTime = time.time()
        framesPerSecond = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(videoFrame, str(int(framesPerSecond)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", videoFrame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
