import cv2
import time
import numpy as np
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def main():
    ##
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    ##
    detector = htm.HandDetector(maxHands=1, detectionCon=0.7)
    pTime = 0
    #fps = 0
    ##
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    volumen_barra = np.interp(volume.GetMasterVolumeLevelScalar(), [0.0, 1.0], [280, 30])
    volumen_porcentaje = int(np.interp(volume.GetMasterVolumeLevelScalar(), [0.0, 1.0], [0, 100]))
    ##

    while True:
        # Capture imag
        exito, imagen = cap.read()
        # Find Hand
        hands, imagen = detector.findHands(imagen, draw = 0)
        lmList, bbox = detector.findPosition(imagen, draw = 0, color=(127,22,69))

        if len(lmList) != 0 and len(bbox) != 0 and hands:
            fingerUplist = detector.fingersUp(hands[0])
            # Filtro basado en tamaño del cuadrado al rededor de la mano
            area = ((bbox[2]-bbox[0]) * (bbox[3]-bbox[1])) // 100
            if 100<area<700:
                detector.drawRectangle(imagen, bbox, txt=hands[0]["type"])
                # Find Distance between index and Thumb
                distanciaRelativa, info, img = detector.findDistance(lmList[4], lmList[8], imagen, color=(255, 0, 0),
                                                                     scale=10)
                if fingerUplist[4] == 0:
                    # Convert Volume
                    volumen_barra = np.interp(distanciaRelativa, [20, 260], [280,30])  # Se convierte la distancia de las landmarks al alto maximo y minimo de la barra
                    volumen_porcentaje = int(np.interp(distanciaRelativa, [20, 260], [0, 100]))  # VLS (0.0,1.0) convert to (0,100)
                    # Reduce Resolution to make it smoother
                    smoothness = 5
                    volumen_porcentaje = smoothness * round(volumen_porcentaje/smoothness)
                    volumen_barra = smoothness * round(volumen_barra/smoothness)
                    # Check finger up
                    fingerUplist = detector.fingersUp(hands[0])
                    # If pinky is down set volume

                    volume.SetMasterVolumeLevelScalar(volumen_porcentaje / 100, None)
                    # Draw

        else:
            volumen_barra = np.interp(volume.GetMasterVolumeLevelScalar(), [0.0, 1.0], [280, 30])
            volumen_porcentaje = int(np.interp(volume.GetMasterVolumeLevelScalar(), [0.0, 1.0], [0, 100]))


        # OPCIONAL
        cv2.putText(imagen, f'VOL: {volumen_porcentaje}', (20, 350), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(imagen, (30,30),(65,280),(0,255,60),3)
        cv2.rectangle(imagen, (30,int(volumen_barra)),(65,280),(0,255,60),cv2.FILLED)

        # Frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        h, w, c = imagen.shape
        (txt_width, txt_height), baseline = cv2.getTextSize(f'FPS: {int(fps)}', cv2.FONT_HERSHEY_COMPLEX, 1, 2)
        x = w - txt_width - 10
        y = h - txt_height - 10
        cv2.putText(imagen, f'FPS: {int(fps)}', (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("PANTALLAAAA", imagen)
        k = cv2.waitKey(1)
        if k == ord("k"):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()