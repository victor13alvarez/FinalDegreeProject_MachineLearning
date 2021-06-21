import cv2
import os


def face_detection(img_path, img_name):
    face_cascade = cv2.CascadeClassifier('face_detector.xml')
    img = cv2.imread(img_path)
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imwrite(os.path.join("ImageResults", img_name), img)
    print('Successfully saved')
