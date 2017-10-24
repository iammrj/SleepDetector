import dlib
from imutils import face_utils
import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
import threading


def resize(img, width, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    # w, h = img.shape
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        resized = imutils.resize(img, width=500)
        return resized


######
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(36, 48):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (a + b) / (2.0 * c)

    # return the eye aspect ratio
    return ear


camera = cv2.VideoCapture("123.mp4")

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
total = 0

while True:
    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=500)
    frame_resized = resize(frame, width=500)

    dets = detector(frame_resized, 1)

    if len(dets) > 0:
        for k, d in enumerate(dets):
            shape = predictor(frame_resized, d)
            shape = shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)

            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame_resized, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame_resized, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.putText(frame_resized, "EAR: {:.3f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if ear < 0.30:
                total += 1
                if total > 18:
                    print("ut jaaaaaaaaaa")
                    cv2.putText(frame_resized, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                total = 0
                print("ttt")

    cv2.imshow("image", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break