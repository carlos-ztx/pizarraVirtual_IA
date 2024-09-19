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

##
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
##

area = 0


while True:
    # Capture imag
    exito, imagen = cap.read()

    # Find Hand
    imagen = detector.findHands(imagen, draw = 0)
    lmList, bbox = detector.findPosition(imagen, draw=False)

    if len(lmList) != 0 and len(bbox) != 0:

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(imagen, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(imagen, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

        # Filter based on size
        area = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) // 100
        if 300<area<1200:
            # Find Distance between index and Thumb
            distanciaRelativa, info, img = detector.findDistance(lmList[4],lmList[8], imagen, color=(255, 0, 0), scale=10)
            # print(f"Distancia entre lm: {distanciaRelativa}")

            # Reduce Resolution to make it smoother
            # Check finger up
            # If pinky is down set volume
            # Draw
            # Frame rate


            # RANGO DE LANDMARKS 20 - 300
            # rango_de_volumen -65 hasta 0

            volumen = np.interp(distanciaRelativa,[15, 260], [minVol, maxVol]) # Se convierte el rango de distancia de las LANDMARKS a VOLUMEN MIN Y MAX
            #print(f"Volumen del sistema: {volumen}")
            volume.SetMasterVolumeLevel(volumen, None)

            volumen_barra = np.interp(distanciaRelativa, [15, 260], [280, 30]) # Se convierte la distancia de las landmarks al alto maximo y minimo de la barra

            # imprimir el volumen como porcentaje de 0 a 100
            volumen_texto = int(np.interp(distanciaRelativa, [15, 260], [0, 100])) # Se convierte la distancia de las landmarks a porcentaje de 0 a 100
            cv2.putText(imagen, f'VOL: {volumen_texto}', (20, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)


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
