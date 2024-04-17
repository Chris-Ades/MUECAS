import mediapipe as mp
import cv2
import time

class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=1, refineLandMarks = False, minDetectionCon=0.5, minTrackCon=0.5):
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
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                0.7, (255, 0, 0), 1)

        return img

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img, True)

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