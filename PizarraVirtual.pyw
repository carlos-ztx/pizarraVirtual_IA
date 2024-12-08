import os
import cv2
import numpy as np
import HandTrackingModule as htm
import time


## Importar las imagenes
carpeta = "Cabecera"
archivos = os.listdir(carpeta)
print(archivos)
lista_cabeceras = []
for nombre in archivos:
    imagen = cv2.imread(f'{carpeta}/{nombre}')
    lista_cabeceras.append(imagen)
print(len(lista_cabeceras))


## Cámara
width, height = 1280, 720
captura = cv2.VideoCapture(0)
captura.set(3, width)
captura.set(4, height)

## Detector de mano en imagen
detector = htm.HandDetector(maxHands=1, detectionCon=0.85, trackCon=0.6)


## Variables
colorPintar, filled= (255, 255, 255), False
grosor_del_pincel = 15
xp, yp = 0, 0
bool_borrador = False
#investigar estooo
canvas_imagen = np.zeros((720,1280,3),np.uint8) # aparentemente crea una matriz bidimensional
tPrevio = 0  # tiempo previo
tActual = 0  # tiempo actual


def Cabecera(imagen, y1=0, y2=125, x1=0, x2=1280, cabecera=(255, 0, 255)):
    """Dibuja en la pantalla lo que enviemos por parametro

        Args:
          imagen: La imagen que queremos modificar.
          y1: Minimo valor de la coordenada y.
          y2: Maximo valor de la coordenada y.
          x1: Minimo valor de la coordenada x.
          x2: Maximo valor de la coordenada x.
          cabecera: Imagen que se superpone.
        """
    if imagen is not None:
        imagen[y1:y2, x1:x2] = cabecera


def actualizarPizarra(imagen):
    ## Mostrar imagen (procesar color para mostrar en pizarra)
    imagen_gris = cv2.cvtColor(canvas_imagen, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imagen_gris, 50, 255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    imagen = cv2.bitwise_and(imagen,imgInv)
    imagen = cv2.bitwise_or(imagen,canvas_imagen)

    ## Otra forma mas de hacerlo
    #imagen = cv2.addWeighted(imagen,0.5,canvas_imagen,0.5,0)

    return imagen

if __name__ == '__main__':
    while True:
        # 1 Importar las imagenes
        exito, imagen = captura.read()
        imagen = cv2.flip(imagen, 1)

        # 2 Encontrar las landmarks
        manos, imagen = detector.findHands(imagen, draw=False)
        lmLista, bbox = detector.findPosition(imagen, draw=False)

        if lmLista:

            # Coordenada de  del dedo indice y medio
            x1, y1 = lmLista[8][1:]
            x2, y2 = lmLista[12][1:]

            # 3 Verificar que y cuantos dedos están levantado
            dedos = detector.fingersUp(manos[0])
            # print(dedos)

            # 4 Dos dedos levantados (modo seleccionar)
            if dedos[1] and dedos[2] and not dedos[0]:
                print("Modo de selección")
                xp, yp = x1, y1
                imagen = actualizarPizarra(imagen)
                Cabecera(imagen, cabecera=lista_cabeceras[0])
                detector.drawRectangle(imagen, bbox, color=colorPintar, txt="Seleccionar")
                # 5.1 Verificar la posicion en el header
                if y1 < 125:
                    if 216 < x1 < 290:
                        Cabecera(imagen, cabecera=lista_cabeceras[1])
                        # imagen[24:98, 216:290] =
                        colorPintar, filled = (255, 255, 255), True
                    elif 342 < x1 < 416:
                        Cabecera(imagen, cabecera=lista_cabeceras[2])
                        colorPintar, filled = (85,113,218), True
                    elif 468 < x1 < 542:
                        Cabecera(imagen, cabecera=lista_cabeceras[3])
                        colorPintar, filled = (89, 222, 255), True
                    elif 594 < x1 < 668:
                        Cabecera(imagen, cabecera=lista_cabeceras[4])
                        colorPintar, filled = (173, 74, 0), True
                    elif 720 < x1 < 794:
                        Cabecera(imagen, cabecera=lista_cabeceras[5])
                        colorPintar, filled = (49, 49, 255), True
                    elif 846 < x1 < 920:
                        Cabecera(imagen, cabecera=lista_cabeceras[6])
                        colorPintar, filled = (87, 217, 126), True
                    elif 972 < x1 < 1046:
                        Cabecera(imagen, cabecera=lista_cabeceras[7])
                        colorPintar, filled = (166, 166, 166), True
                    elif 1099 < x1 < 1220:  # Borrador
                        Cabecera(imagen, cabecera=lista_cabeceras[8])
                        colorPintar, filled, bool_borrador = (0, 0, 0), False, True
                    else:  # Default
                        Cabecera(imagen, cabecera=lista_cabeceras[0])

            # 5 Un dedo levantado - Dedo indice dibuja
            elif dedos[1] and not dedos[2] and not dedos[0]:
                print("Modo de dibujar")
                detector.drawRectangle(imagen, bbox, color=(121, 237, 12), txt="Dibujar")
                imagen = actualizarPizarra(imagen)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if filled:
                    cv2.circle(imagen, (x1, y1), 15, colorPintar, cv2.FILLED)
                    # cv2.line(imagen,(xp,yp),(x1,y1),colorPintar,grosor_del_pincel)
                    cv2.line(canvas_imagen, (xp, yp), (x1, y1), colorPintar, grosor_del_pincel)
                    xp, yp = x1, y1

                if not filled and bool_borrador:
                    cv2.circle(imagen, (x1, y1), 15, (255, 255, 255))
                    # cv2.line(imagen,(xp,yp),(x1,y1),colorPintar,grosor_del_pincel)
                    cv2.line(canvas_imagen, (xp, yp), (x1, y1), colorPintar, 60)
                    xp, yp = x1, y1
            elif dedos[1] and dedos[2] and dedos[0]:
                canvas_imagen = np.zeros((720,1280,3),np.uint8)
            else:
                imagen = actualizarPizarra(imagen)
        else:
            imagen = actualizarPizarra(imagen)

        # imprime FPS
        tActual = time.time()
        fps = 1 / (tActual - tPrevio)
        tPrevio = tActual
        cv2.putText(imagen, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow("Pizarra", imagen)
        cv2.imshow("Canvas", canvas_imagen)

        k = cv2.waitKey(1)
        if k == ord("k"):
            captura.release()
            cv2.destroyAllWindows()
            break

