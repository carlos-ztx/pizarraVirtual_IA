import math
import time
import cv2
import mediapipe as mp


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



    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # mpDraw.draw_landmarks(img, handLms) #esta linea dibuja los puntos en la mano
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # esta linea dibuja puntos y lineas
        return img


    def findPosition(self, img, handNumber=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # se obtiene la altura, ancho y profundidad de la imagen
                cx, cy = int(lm.x * w), int(lm.y * h)  # se multiplica lass coordenadas de los landmark con la altura y anchura de la imagen
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (254, 0, 0, 0), cv2.FILLED)

        return lmList


    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED) #p1
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED) #p2
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
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])

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
