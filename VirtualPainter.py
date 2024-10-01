import os

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info and warning messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to error only

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

import cv2
import numpy as np
import HandTrackingModule as htm
import math

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

headerDirectory = "Header"
headerFiles = os.listdir(headerDirectory)
headerImages = []
penThickness = 6
eraserSize = 110
prevX, prevY = 0, 0

# Variables to store fixed sizes
fixedPenThickness = penThickness
fixedEraserSize = eraserSize

for imageFile in headerFiles:
    image = cv2.imread(f'{headerDirectory}/{imageFile}')
    headerImages.append(image)

currentHeader = headerImages[0]
currentColor = (255, 0, 0)  # Default color for gauge is red initially

cameraFeed = cv2.VideoCapture(0)
cameraFeed.set(3, 1280)
cameraFeed.set(4, 720)

if not cameraFeed.isOpened():
    print("Error: Could not open video capture.")
    exit()

handTracker = htm.HandTracker(detectionCon=0.85)

canvasImage = np.zeros((720, 1280, 3), np.uint8)

try:
    while True:
        frameSuccess, frame = cameraFeed.read()
        if not frameSuccess:
            print("Failed to read frame from camera. Exiting...")
            break

        frame = cv2.flip(frame, 1)

        frame = handTracker.findHands(frame)
        landmarkList = handTracker.findPosition(frame, draw=False)

        if len(landmarkList) != 0:
            # Thumb (landmark 4) and Index Finger Tip (landmark 8) positions
            thumbX, thumbY = landmarkList[4][1:]
            indexX, indexY = landmarkList[8][1:]
            middleX, middleY = landmarkList[12][1:]

            # Calculate the distance between the index finger and thumb
            length = math.hypot(indexX - thumbX, indexY - thumbY)

            # Checking which fingers are up
            fingerStates = handTracker.fingersUp()

            # Condition to adjust thickness (only thumb and index finger are up)
            if fingerStates[0] == 1 and fingerStates[1] == 1 and all(finger == 0 for finger in fingerStates[2:]):
                # Adjust pen thickness and eraser size based on the distance
                fixedPenThickness = int(np.interp(length, [30, 200], [6, 25]))  # Pen thickness
                fixedEraserSize = int(np.interp(length, [30, 200], [50, 200]))  # Eraser size
                sizePercentage = int(np.interp(length, [30, 200], [0, 100]))  # Size percentage

                # Draw a line between the thumb and index finger
                cv2.line(frame, (thumbX, thumbY), (indexX, indexY), (255, 0, 255), 3)

                # Display size meter halfway between the toolbar and the bottom of the screen on the right side
                cv2.rectangle(frame, (1195, 297), (1230, 547), currentColor, 3)
                cv2.rectangle(frame, (1195, int(547 - (sizePercentage * 2.5))), (1230, 547), currentColor, cv2.FILLED)
                cv2.putText(frame, f'{sizePercentage} %', (1185, 597), cv2.FONT_HERSHEY_COMPLEX, 1, currentColor, 3)

            # Selection mode - index and middle fingers are up
            elif fingerStates[1] == 1 and fingerStates[2] == 1:
                prevX, prevY = 0, 0

                # Draw a point between index and middle fingers
                midX, midY = (indexX + middleX) // 2, (indexY + middleY) // 2
                cv2.circle(frame, (midX, midY), 15, currentColor, cv2.FILLED)

                # Check if the selection is in the header area
                if indexY < 125:
                    if 100 < indexX < 240:
                        currentHeader = headerImages[0]
                        currentColor = (255, 0, 120)
                    elif 280 < indexX < 410:
                        currentHeader = headerImages[1]
                        currentColor = (0, 0, 255)
                    elif 430 < indexX < 590:
                        currentHeader = headerImages[2]
                        currentColor = (255, 0, 0)
                    elif 630 < indexX < 770:
                        currentHeader = headerImages[3]
                        currentColor = (0, 255, 255)
                    elif 790 < indexX < 920:
                        currentHeader = headerImages[4]
                        currentColor = (0, 255, 0)
                    elif 1050 < indexX < 1180:
                        currentHeader = headerImages[5]
                        currentColor = (0, 0, 0)

            # Drawing mode - only index finger is up
            elif fingerStates[1] == 1 and all(finger == 0 for finger in fingerStates[2:]):
                cv2.circle(frame, (indexX, indexY), 15, currentColor, cv2.FILLED)

                if prevX == 0 and prevY == 0:
                    prevX, prevY = indexX, indexY

                if currentColor == (0, 0, 0):  # Eraser mode
                    cv2.line(frame, (prevX, prevY), (indexX, indexY), currentColor, fixedEraserSize)
                    cv2.line(canvasImage, (prevX, prevY), (indexX, indexY), currentColor, fixedEraserSize)
                else:  # Drawing mode
                    cv2.line(frame, (prevX, prevY), (indexX, indexY), currentColor, fixedPenThickness)
                    cv2.line(canvasImage, (prevX, prevY), (indexX, indexY), currentColor, fixedPenThickness)

                prevX, prevY = indexX, indexY

        try:
            canvasGray = cv2.cvtColor(canvasImage, cv2.COLOR_BGR2GRAY)
            _, canvasInverse = cv2.threshold(canvasGray, 50, 255, cv2.THRESH_BINARY_INV)
            canvasInverse = cv2.cvtColor(canvasInverse, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, canvasInverse)
            frame = cv2.bitwise_or(frame, canvasImage)
        except Exception as e:
            print(f"Error during image processing: {e}")
            break

        frame[0:125, 0:1280] = currentHeader

        cv2.imshow("Image", frame)
        cv2.imshow("Canvas", canvasImage)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")
finally:
    cameraFeed.release()
    cv2.destroyAllWindows()
