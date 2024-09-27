import os
import warnings
import cv2
import numpy as np
import HandTrackingModule as htm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

headerDirectory = "Header"
headerFiles = os.listdir(headerDirectory)
headerImages = []
penThickness = 6
eraserSize = 110
prevX, prevY = 0, 0

for imageFile in headerFiles:
    image = cv2.imread(f'{headerDirectory}/{imageFile}')
    headerImages.append(image)

currentHeader = headerImages[0]
currentColor = (0, 0, 0)

cameraFeed = cv2.VideoCapture(0)
cameraFeed.set(3, 1280)
cameraFeed.set(4, 720)

# Check if the camera opened successfully
if not cameraFeed.isOpened():
    print("Error: Could not open video capture.")
    exit()

handTracker = htm.HandTracker(detectionCon=0.85)

canvasImage = np.zeros((720, 1280, 3), np.uint8)

try:
    while True:
        # 1. Import the image
        frameSuccess, frame = cameraFeed.read()
        if not frameSuccess:
            print("Failed to read frame from camera. Exiting...")
            break

        frame = cv2.flip(frame, 1)

        # 2. Find hand landmarks
        frame = handTracker.findHands(frame)
        landmarkList = handTracker.findPosition(frame, draw=False)

        if len(landmarkList) != 0:
            # Tip of index and middle fingers
            indexX, indexY = landmarkList[8][1:]
            middleX, middleY = landmarkList[12][1:]

            # 3. Checking which fingers are up
            fingerStates = handTracker.fingersUp()
            print(fingerStates)

            # 4. If selection mode = two fingers are up
            if fingerStates[1] and fingerStates[2]:
                prevX, prevY = 0, 0
                print("Selection Mode")
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

                cv2.rectangle(frame, (indexX, indexY - 25), (middleX, middleY + 25), currentColor, cv2.FILLED)

            # 5. If drawing mode - index finger is up
            elif fingerStates[1] and not fingerStates[2]:
                cv2.circle(frame, (indexX, indexY), 15, currentColor, cv2.FILLED)
                print("Drawing Mode")
                if prevX == 0 and prevY == 0:
                    prevX, prevY = indexX, indexY

                if currentColor == (0, 0, 0):
                    cv2.line(frame, (prevX, prevY), (indexX, indexY), currentColor, eraserSize)
                    cv2.line(canvasImage, (prevX, prevY), (indexX, indexY), currentColor, eraserSize)
                else:
                    cv2.line(frame, (prevX, prevY), (indexX, indexY), currentColor, penThickness)
                    cv2.line(canvasImage, (prevX, prevY), (indexX, indexY), currentColor, penThickness)

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

        # Setting the header image
        frame[0:125, 0:1280] = currentHeader

        cv2.imshow("Image", frame)
        cv2.imshow("Canvas", canvasImage)

        # Adding delay and proper exit condition
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")
finally:
    cameraFeed.release()
    cv2.destroyAllWindows()
