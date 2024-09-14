import cv2
import time
import numpy as np
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


##
wCam, hCam = 640, 480
##

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0

detector = htm.HandDetector(maxHands=1, detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()k
rango_de_volumen = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-5, None)
minVol = rango_de_volumen[0]
maxVol = rango_de_volumen[1]
volumen = 0
volumen_barra = 280


while True:
    exito, imagen = cap.read()
    imagen = detector.findHands(imagen, draw = 0)
    lmList = detector.findPosition(imagen, draw=False)
    if len(lmList) != 0:
        #print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1],lmList[4][2]
        x2, y2 = lmList[8][1],lmList[8][2]

        cv2.circle(imagen,(x1,y1),15,(255,0,255),cv2.FILLED)
        cv2.circle(imagen,(x2,y2),15,(255,0,255),cv2.FILLED)

        distanciaRelativa, info, img = detector.findDistance((x1,y1), (x2, y2), imagen, color=(255, 0, 0), scale=10)
        print(distanciaRelativa)



        # RANGO DE LANDMARKS 20 - 300
        # rango_de_volumen -65 hasta 0

        volumen = np.interp(distanciaRelativa,[20, 300], [minVol, maxVol])
        print(volumen)
        volume.SetMasterVolumeLevel(volumen, None)

        volumen_barra = np.interp(distanciaRelativa, [20, 300], [280, 30])


     # OPCIONAL
    cv2.rectangle(imagen, (30,30),(65,280),(0,255,60),3)
    cv2.rectangle(imagen, (30,int(volumen_barra)),(65,280),(0,255,60),cv2.FILLED)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime
    #cv2.putText(imagen,f'FPS: {int(fps)}',(35, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("PANTALLAAAA", imagen)
    k = cv2.waitKey(1)
    if k == ord("k"):
        break

cap.release()
cv2.destroyAllWindows()
