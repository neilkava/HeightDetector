from cv2 import MARKER_CROSS
import mediapipe as mp
import cv2
import time

mpSol = mp.solutions
mpPose = mpSol.pose
pose = mpPose.Pose()
mpFaceMesh = mpSol.face_mesh
facemesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
mpDraw = mpSol.drawing_utils
drawing = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)
capture = cv2.VideoCapture(0)
list = []
n = 0
# ptime = 0

while True:
    isTrue, img = capture.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            list[n] = list.append([id, lm.x, lm.y])
            n + 1
            h, w, c = img.shape
            if id == 32:
                cx1, cy1 = int(lm.x * w), int(lm.y * h)
                cv2.drawMarker(img, (cx1, cy1), (255, 255, 255), MARKER_CROSS, 10, 2)
                d = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5
                di = round(d * 0.5)

                cv2.putText(img, "Height: ", (25, 35), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), thickness = 2)
                cv2.putText(img, str(di), (185, 35), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), thickness = 2)
                cv2.putText(img, "cm", (260, 35), cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 0, 0), thickness = 2)

                if ord('q'):
                    cv2.destroyAllWindows()
                    break
            if id == 6:
                cx2 , cy2 = int(lm.x * w) , int(lm.y * h)
                cy2 = cy2 + 20
                cv2.drawMarker(img, (cx2, cy2), (255, 255, 255), MARKER_CROSS, 10, 2)
    img = cv2.resize(img , (700, 500))

    cv2.imshow("Height Detector", img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    time.sleep(1)

capture.release()
cv2.destroyAllWindows()