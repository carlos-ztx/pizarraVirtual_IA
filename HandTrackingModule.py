import math
import time
import cv2
import mediapipe as mp
"""
Montar presentación con el proyecto
"""

class HandDetector():
    def __init__(self, mode=False, maxHands=2, complex=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complex = complex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, drawRectangle = False, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                x_min, x_max = min(xList), max(xList)
                y_min, y_max = min(yList), max(yList)
                bbox = x_min, y_min, x_max, y_max
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                if drawRectangle:
                    self.drawRectangle(img, bbox,txt = myHand["type"])


        return allHands, img


    def findPosition(self, img, handNumber = 0, color = (254, 0, 0, 0), draw=True):
        lista_x = []
        lista_y = []
        bbox = []
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # se obtiene la altura, ancho y profundidad (creo) de la imagen
                cx, cy = int(lm.x * w), int(lm.y * h)  # se multiplica lass coordenadas de los landmark con la altura y anchura de la imagen
                lmlist.append([id, cx, cy])
                lista_x.append(cx)
                lista_y.append(cy)
                if draw:
                    cv2.circle(img, (cx, cy), 7, color, cv2.FILLED)

        # Bounding box info
        if len(lista_x) != 0 and len(lista_y) != 0:
            x_min, x_max = min(lista_x), max(lista_x)
            y_min, y_max = min(lista_y), max(lista_y)
            bbox = x_min, y_min, x_max, y_max

        return lmlist, bbox

    def drawRectangle(self,img, bbox,color=(255, 0, 255), txt = ""):
        if img is not None and len(bbox) !=0:
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), color, 2)
            cv2.putText(img, txt, (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:

            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5, draw = True):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if draw:
            cv2.circle(img, (x1, y1), 5, color, cv2.FILLED) #p1
            cv2.circle(img, (x2, y2), 5, color, cv2.FILLED) #p2
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3)) #linea entre p1 y p2
            cv2.circle(img, (cx, cy), 3, (120,50, 125), cv2.FILLED) #pCentro

        return length, info, img


def main():

    pTime = 0  # previus time
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow("Imagen", img)
        k = cv2.waitKey(1)
        if k == ord("k"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()