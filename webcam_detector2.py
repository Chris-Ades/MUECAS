import mediapipe as mp
import cv2
import time
import math
import numpy as np

class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=1, refineLandMarks = True, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandMarks = refineLandMarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refineLandMarks, self.minDetectionCon, self. minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
    

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                self.drawSpec, self.drawSpec)

                for id, lm in enumerate(faceLms.landmark):
                    if id in [159, 145, 374, 386]:
                        ih, iw, ic = img.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    0.6, (255, 0, 0), 1)

                lmList = []
                ih, iw, ic = img.shape
                for lms in faceLms.landmark:
                    lmList.append((lms.x*iw, lms.y*ih))

                # North South East West Tilts
                # x_nose_tip, y_nose_tip = lmList[1]
                # x_right_eye_inner, y_right_eye_inner = lmList[133]
                # x_left_eye_inner, y_left_eye_inner = lmList[362]

                # XY Rotation

                # Z direction
                x108, y108 = lmList[108]
                x337, y337 = lmList[337]
                distance = math.hypot(x108 - x337, y108 - y337)
                # Want range: 0 to 100
                # Actual range: 20 to 90
                distance_Z = np.interp(distance, [20, 90], [0, 100])
                cv2.putText(img, f'Z: {distance_Z:.2f}', (20, 190), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)
                
                x, y = lmList[1]
                # X direction
                # # X Range is [150, 500] at distance_Z = 100 with minmax [20, 90]
                # # X Range is [40, 600] at distance_Z = 0  with minmax [20, 90]

                current_x_max = np.interp(distance_Z, [0, 100], [600, 500])
                current_x_min = np.interp(distance_Z, [0, 100], [40, 150])
                x = np.interp(x, [current_x_min, current_x_max], [0, 100])
                # x = np.interp(distance, [current_x_min, current_x_max], [0, 100])
                cv2.putText(img, f'X: {x:.2f}', (20, 130), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)

                # Y direction
                # # Y Range is [315, 220] at distance_Z = 100 with Z actual range [20, 90]
                # # Y Range is [423, 60] at distance_Z = 0  with Z actual range [20, 90]
                current_y_max = np.interp(distance_Z, [0, 100], [423, 315])
                current_y_min = np.interp(distance_Z, [0, 100], [60, 220])
                y = np.interp(y, [current_y_min, current_y_max], [100, 0])
                cv2.putText(img, f'Y: {y:.2f}', (20, 160), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)

                # Mouth Vertical
                x13, y13 = lmList[13]
                x14, y14 = lmList[14]
                distance = math.hypot(x13 - x14, y13 - y14)
                # Vertical Mouth Range is [1, 90] at distance_Z = 100 with minmax [20, 90]
                # Vertical Mouth Range is [1, 24] at distance_Z = 0  with minmax [20, 90]
                current_mouth_vert_max = np.interp(distance_Z, [0, 100], [24, 90])
                mouth_vert = np.interp(distance, [2, current_mouth_vert_max], [0, 100])
                cv2.putText(img, f'Mouth Vertical: {mouth_vert:.2f}', (20, 220), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)

                # Mouth Horizontal
                x78, y78 = lmList[78]
                x308, y308 = lmList[308]
                distance = math.hypot(x78 - x308, y78 - y308)
                # Horizontal Mouth Range is [90, 180] at distance_Z = 100 with minmax [20, 90]
                # Horizontal Mouth Range is [18, 33] at distance_Z = 0  with minmax [20, 90]
                current_mouth_horiz_max = np.interp(distance_Z, [0, 100], [33, 180])
                current_mouth_horiz_min = np.interp(distance_Z, [0, 100], [18, 90])
                mouth_horiz = np.interp(distance, [current_mouth_horiz_min, current_mouth_horiz_max], [0, 100])
                cv2.putText(img, f'Mouth Horizontal: {mouth_horiz:.2f}', (20, 250), cv2.FONT_HERSHEY_PLAIN,
                             1, (0, 0, 0), 2)

                # Right Eye Blinks
                x145, y145 = lmList[145]
                x159, y159 = lmList[159]
                distance = math.hypot(x145 - x159, y145 - y159)
                # Right Eye Range is [10, 28] at distance_Z = 100 with minmax [20, 90]
                # Right Eye Range is [3, 5] at distance_Z = 0  with minmax [20, 90]
                current_right_eye_max = np.interp(distance_Z, [0, 100], [5, 28])
                current_right_eye_min = np.interp(distance_Z, [0, 100], [3, 10])
                right_eye = np.interp(distance, [current_right_eye_min, current_right_eye_max], [0, 100])
 
                # Left Eye
                x374, y374 = lmList[374]
                x386, y386 = lmList[386]
                distance = math.hypot(x374 - x386, y374 - y386)
                # left Eye Range is [10, 28] at distance_Z = 100 with minmax [20, 90]
                # left Eye Range is [3, 5] at distance_Z = 0  with minmax [20, 90]
                current_left_eye_max = np.interp(distance_Z, [0, 100], [5, 28])
                current_left_eye_min = np.interp(distance_Z, [0, 100], [3, 10])
                left_eye = np.interp(distance, [current_left_eye_min, current_left_eye_max], [0, 100])

                eyes = left_eye + right_eye
                if eyes > 100:
                    eyes=1
                else:
                    eyes=0
                cv2.putText(img, f'Eyes Open: {int(eyes)}', (20, 280), cv2.FONT_HERSHEY_PLAIN,
                             1, (0, 0, 0), 2)
                    
                # Eyebrows
                

        return img

def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img, True)
        #img = cv2.flip(img, 1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("MUECAS", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()