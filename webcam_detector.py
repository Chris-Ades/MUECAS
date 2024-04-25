import mediapipe as mp
import cv2
import math
import numpy as np
from pyo import *

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
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        self.z = 0 
        self.x = 0
        self.y = 0
        self.mouth_vert = 0
        self.mouth_horiz = 0
        self.blink = 0
        self.eyebrows = 0
    

    def findFaceMesh(self, img, show_landmarks=False, show_id=False):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if show_landmarks:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                self.drawSpec, self.drawSpec)

                if show_id:
                    for id, lm in enumerate(faceLms.landmark):
                        #if id in []:
                        ih, iw, ic = img.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                        cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    0.6, (255, 0, 0), 1)

                lmList = []
                ih, iw, ic = img.shape
                for lms in faceLms.landmark:
                    lmList.append((lms.x*iw, lms.y*ih))

                # Z direction
                x108, y108 = lmList[108]
                x337, y337 = lmList[337]
                distance = math.hypot(x108 - x337, y108 - y337)
                # Want range: 0 to 100
                # Actual range: 20 to 90
                self.z = np.interp(distance, [20, 90], [0, 100])
                cv2.putText(img, f'Z: {self.z:.2f}', (20, 190), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)
                
                x, y = lmList[1]

                # X direction
                # # X Range is [150, 500] at self.z = 100 with minmax [20, 90]
                # # X Range is [40, 600] at self.z = 0  with minmax [20, 90]
                current_x_max = np.interp(self.z, [0, 100], [600, 500])
                current_x_min = np.interp(self.z, [0, 100], [40, 150])
                self.x = np.interp(x, [current_x_min, current_x_max], [0, 100])
                cv2.putText(img, f'X: {self.x:.2f}', (20, 130), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)

                # Y direction
                # # Y Range is [315, 220] at self.z = 100 with minmax [20, 90]
                # # Y Range is [423, 60] at self.z = 0  with minmax [20, 90]
                current_y_max = np.interp(self.z, [0, 100], [423, 315])
                current_y_min = np.interp(self.z, [0, 100], [60, 220])
                self.y = np.interp(y, [current_y_min, current_y_max], [100, 0])
                cv2.putText(img, f'Y: {self.y:.2f}', (20, 160), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)

                # Mouth Vertical
                x13, y13 = lmList[13]
                x14, y14 = lmList[14]
                distance = math.hypot(x13 - x14, y13 - y14)
                # Vertical Mouth Range is [1, 90] at self.z = 100 with minmax [20, 90]
                # Vertical Mouth Range is [1, 24] at self.z = 0  with minmax [20, 90]
                current_mouth_vert_max = np.interp(self.z, [0, 100], [24, 90])
                self.mouth_vert = np.interp(distance, [2, current_mouth_vert_max], [0, 100])
                cv2.putText(img, f'Mouth Vertical: {self.mouth_vert:.2f}', (20, 220), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 0), 2)

                # Mouth Horizontal
                x78, y78 = lmList[78]
                x308, y308 = lmList[308]
                distance = math.hypot(x78 - x308, y78 - y308)
                # Horizontal Mouth Range is [90, 180] at self.z = 100 with minmax [20, 90]
                # Horizontal Mouth Range is [18, 33] at self.z = 0  with minmax [20, 90]
                current_mouth_horiz_max = np.interp(self.z, [0, 100], [33, 180])
                current_mouth_horiz_min = np.interp(self.z, [0, 100], [18, 90])
                self.mouth_horiz = np.interp(distance, [current_mouth_horiz_min, current_mouth_horiz_max], [0, 100])
                cv2.putText(img, f'Mouth Horizontal: {self.mouth_horiz:.2f}', (20, 250), cv2.FONT_HERSHEY_PLAIN,
                             1, (0, 0, 0), 2)

                # Right Eye
                x145, y145 = lmList[145]
                x159, y159 = lmList[159]
                distance = math.hypot(x145 - x159, y145 - y159)
                # Right Eye Range is [10, 28] at self.z = 100 with minmax [20, 90]
                # Right Eye Range is [3, 5] at self.z = 0  with minmax [20, 90]
                current_right_eye_max = np.interp(self.z, [0, 100], [5, 28])
                current_right_eye_min = np.interp(self.z, [0, 100], [3, 10])
                right_eye = np.interp(distance, [current_right_eye_min, current_right_eye_max], [0, 100])
                # Left Eye
                x374, y374 = lmList[374]
                x386, y386 = lmList[386]
                distance = math.hypot(x374 - x386, y374 - y386)
                # Left Eye Range is [10, 28] at self.z = 100 with minmax [20, 90]
                # Left Eye Range is [3, 5] at self.z = 0  with minmax [20, 90]
                current_left_eye_max = np.interp(self.z, [0, 100], [5, 28])
                current_left_eye_min = np.interp(self.z, [0, 100], [3, 10])
                left_eye = np.interp(distance, [current_left_eye_min, current_left_eye_max], [0, 100])
                eyes = left_eye + right_eye
                if eyes < 100:
                    self.blink=1
                else:
                    self.blink=0
                cv2.putText(img, f'Blink: {int(self.blink)}', (20, 280), cv2.FONT_HERSHEY_PLAIN,
                             1, (0, 0, 0), 2)

                # Eyebrows
                x66, y66 = lmList[66]
                x69, y69 = lmList[69]
                distance = math.hypot(x66 - x69, y66 - y69)
                x296, y296 = lmList[296]
                x299, y299 = lmList[299]
                distance = distance + math.hypot(x296 - x299, y296 - y299)
                # Left Eye Range is [65, 25] at self.z = 100 with minmax [20, 90]
                # Left Eye Range is [15, 11] at self.z = 0  with minmax [20, 90]
                current_eyebrows_max = np.interp(self.z, [0, 100], [11, 25])
                current_eyebrows_min = np.interp(self.z, [0, 100], [15, 65])
                self.eyebrows = np.interp(distance, [current_eyebrows_max, current_eyebrows_min], [100, 0])
                cv2.putText(img, f'Eyebrows: {self.eyebrows:.2f}', (20, 310), cv2.FONT_HERSHEY_PLAIN,
                             1, (0, 0, 0), 2)

                # XY Rotation

        return img 

def main():
    def pixelate(image, pixel_size):
        height, width = image.shape[:2]
        small_image = cv2.resize(image, (pixel_size, pixel_size), interpolation=cv2.INTER_NEAREST)

        if pixel_size < 23:
            num_recolor = pixel_size*pixel_size // 6
            random_rows = np.random.randint(0, height, num_recolor)
            random_cols = np.random.randint(0, width, num_recolor)
            random_colors = [image[row, col] for row, col in zip(random_rows, random_cols)]
            
            # Recolor the selected pixels
            for i in range(num_recolor):
                row = random_rows[i] % pixel_size  # Ensure row index is within bounds
                col = random_cols[i] % pixel_size  # Ensure column index is within bounds
                small_image[row, col] = random_colors[i]

        pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)


        return pixelated_image

    # Video
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()

    # Sound
    pyo_server = Server().boot().start()
    sender = OscDataSend(types="iffffff", port=9900, address="/face_data", host="192.168.9.223")
    
    blink = 0
    x = 0
    y = 0
    z = 0
    mouth_horiz = 0
    mouth_vert = 0
    eyebrows = 0

    def send():
        sender.send([blink, x, y, z, mouth_horiz, mouth_vert, eyebrows])

    pat = Pattern(send, time=0.001).play()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findFaceMesh(img, show_landmarks=False)
        
        
        blink = detector.blink
        x = detector.x
        y = detector.y
        z = detector.z
        mouth_horiz = detector.mouth_horiz
        mouth_vert = detector.mouth_vert
        eyebrows = detector.eyebrows

        if blink:
            img = 255 - img

        if y <= 50:
            img = pixelate(img, pixel_size=int((np.interp(y, [0, 50], [20, 1000])).item()))

        cv2.imshow("Muecas", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pyo_server.shutdown()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()